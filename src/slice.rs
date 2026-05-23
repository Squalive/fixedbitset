use crate::{
    Block, Difference, IndexRange, Intersection, Masks, Ones, SimdBlock, SymmetricDifference,
    Union, Zeroes, div_rem,
};
use core::{
    cmp::Ordering,
    fmt::{Binary, Display, Error, Formatter, Write},
    marker::PhantomData,
    ptr::NonNull,
};

#[derive(Debug, Clone, Copy)]
pub struct FixedBitCell<'a> {
    data: NonNull<Block>,
    bits: usize,
    marker: PhantomData<&'a mut [Block]>,
}

// SAFETY: FixedBitSlice does not contains thread-local state
unsafe impl<'a> Send for FixedBitCell<'a> {}

// SAFETY: No mutable access is allowed in this
unsafe impl<'a> Sync for FixedBitCell<'a> {}

impl<'a> FixedBitCell<'a> {
    /// # Safety
    /// - data must be a valid ptr.
    /// - blocks must be a valid range in data
    pub unsafe fn from_raw_parts(data: *const Block, bits: usize) -> FixedBitCell<'a> {
        FixedBitCell {
            data: NonNull::new_unchecked(data.cast_mut()),
            bits,
            marker: PhantomData,
        }
    }

    /// # Safety
    /// Caller must ensures subblock is inside the range of blocks.len()
    #[inline]
    unsafe fn get_unchecked(self, subblock: usize) -> &'a Block {
        // SAFETY: caller ensures the safety of this
        unsafe { &*self.data.as_ptr().cast::<Block>().add(subblock) }
    }

    fn block_len(self) -> usize {
        self.bits.div_ceil(Block::BITS as usize)
    }

    fn simd_block_len(self) -> usize {
        self.bits.div_ceil(SimdBlock::BITS)
    }

    fn as_simd_slice(self) -> &'a [SimdBlock] {
        // SAFETY: we are within the range since Block is multiple of SimdBlock
        unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast(), self.simd_block_len()) }
    }

    #[inline]
    fn batch_count_ones(blocks: impl IntoIterator<Item = Block>) -> usize {
        blocks.into_iter().map(|x| x.count_ones() as usize).sum()
    }

    /// View the bitset as a slice of `Block` blocks
    #[inline]
    pub fn as_slice(self) -> &'a [Block] {
        // SAFETY: The bits from both usize and Block are required to be reinterprettable, and
        // neither have any padding or alignment issues. The slice constructed is within bounds
        // of the underlying allocation. This function is called with a read-only  borrow so
        // no other write can happen as long as the returned borrow lives.
        unsafe { core::slice::from_raw_parts(self.data.as_ptr(), self.block_len()) }
    }

    #[inline]
    pub fn len(self) -> usize {
        self.bits
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.bits == 0
    }

    #[inline]
    pub fn is_clear(self) -> bool {
        self.as_simd_slice().iter().all(|block| block.is_empty())
    }

    /// Finds the lowest set bit in the bitset.
    ///
    /// Returns `None` if there aren't any set bits.
    #[inline]
    pub fn minimum(self) -> Option<usize> {
        let (block_idx, block) = self
            .as_simd_slice()
            .iter()
            .enumerate()
            .find(|&(_, block)| !block.is_empty())?;
        let mut inner = 0;
        let mut trailing = 0;
        for subblock in block.into_usize_array() {
            if subblock != 0 {
                trailing = subblock.trailing_zeros() as usize;
                break;
            } else {
                inner += Block::BITS as usize;
            }
        }
        Some(block_idx * SimdBlock::BITS + inner + trailing)
    }

    /// Finds the highest set bit in the bitset.
    ///
    /// Returns `None` if there aren't any set bits.
    #[inline]
    pub fn maximum(self) -> Option<usize> {
        let (block_idx, block) = self
            .as_simd_slice()
            .iter()
            .rev()
            .enumerate()
            .find(|&(_, block)| !block.is_empty())?;
        let mut inner = 0;
        let mut leading = 0;
        for subblock in block.into_usize_array().iter().rev() {
            if *subblock != 0 {
                leading = subblock.leading_zeros() as usize;
                break;
            } else {
                inner += Block::BITS as usize;
            }
        }
        let max = self.simd_block_len() * SimdBlock::BITS;
        Some(max - block_idx * SimdBlock::BITS - inner - leading - 1)
    }

    /// `true` if all bits in the [`FixedBitSet`] are set.
    ///
    /// This is equivalent to [`bitset.count_ones(..) == bitset.len()`](FixedBitSet::count_ones).
    #[inline]
    pub fn is_full(self) -> bool {
        self.contains_all_in_range(..)
    }

    /// Return **true** if the bit is enabled in the **FixedBitSet**,
    /// **false** otherwise.
    ///
    /// Note: bits outside the capacity are always disabled.
    ///
    /// Note: Also available with index syntax: `bitset[bit]`.
    #[inline]
    pub fn contains(self, bit: usize) -> bool {
        if bit < self.bits {
            // SAFETY: The above check ensures that the block and bit are within bounds.
            unsafe { self.contains_unchecked(bit) }
        } else {
            false
        }
    }

    /// Return **true** if the bit is enabled in the **FixedBitSet**,
    /// **false** otherwise.
    ///
    /// Note: unlike `contains`, calling this with an invalid `bit`
    /// is undefined behavior.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn contains_unchecked(self, bit: usize) -> bool {
        let (block, i) = div_rem(bit, Block::BITS as usize);
        // SAFETY: Caller ensures safety
        (unsafe { self.get_unchecked(block) } & (1 << i)) != 0
    }

    /// Checks if the bitset contains every bit in the given range.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn contains_all_in_range<T: IndexRange>(self, range: T) -> bool {
        for (block, mask) in Masks::new(range, self.bits) {
            // SAFETY: Masks cannot return a block index that is out of range.
            let block = unsafe { self.get_unchecked(block) };
            if block & mask != mask {
                return false;
            }
        }
        true
    }

    /// Checks if the bitset contains at least one set bit in the given range.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn contains_any_in_range<T: IndexRange>(self, range: T) -> bool {
        for (block, mask) in Masks::new(range, self.bits) {
            // SAFETY: Masks cannot return a block index that is out of range.
            let block = unsafe { self.get_unchecked(block) };
            if block & mask != 0 {
                return true;
            }
        }
        false
    }

    /// Count the number of set bits in the given bit range.
    ///
    /// This function is potentially much faster than using `ones(other).count()`.
    /// Use `..` to count the whole content of the bitset.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn count_ones<T: IndexRange>(self, range: T) -> usize {
        Self::batch_count_ones(Masks::new(range, self.bits).map(|(block, mask)| {
            // SAFETY: Masks cannot return a block index that is out of range.
            unsafe { *self.get_unchecked(block) & mask }
        }))
    }

    /// Count the number of unset bits in the given bit range.
    ///
    /// This function is potentially much faster than using `zeroes(other).count()`.
    /// Use `..` to count the whole content of the bitset.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn count_zeroes<T: IndexRange>(self, range: T) -> usize {
        Self::batch_count_ones(Masks::new(range, self.bits).map(|(block, mask)| {
            // SAFETY: Masks cannot return a block index that is out of range.
            unsafe { !*self.get_unchecked(block) & mask }
        }))
    }

    /// Iterates over all enabled bits.
    ///
    /// Iterator element is the index of the `1` bit, type `usize`.
    #[inline]
    pub fn ones(self) -> Ones<'a> {
        match self.as_slice().split_first() {
            Some((&first_block, rem)) => {
                let (&last_block, rem) = rem.split_last().unwrap_or((&0, rem));
                Ones {
                    bitset_front: first_block,
                    bitset_back: last_block,
                    block_idx_front: 0,
                    block_idx_back: (1 + rem.len()) * Block::BITS as usize,
                    remaining_blocks: rem.iter(),
                }
            }
            None => Ones {
                bitset_front: 0,
                bitset_back: 0,
                block_idx_front: 0,
                block_idx_back: 0,
                remaining_blocks: [].iter(),
            },
        }
    }

    /// Iterates over all disabled bits.
    ///
    /// Iterator element is the index of the `0` bit, type `usize`.
    #[inline]
    pub fn zeroes(self) -> Zeroes<'a> {
        match self.as_slice().split_first() {
            Some((&block, rem)) => Zeroes {
                bitset: !block,
                block_idx: 0,
                len: self.len(),
                remaining_blocks: rem.iter(),
            },
            None => Zeroes {
                bitset: !0,
                block_idx: 0,
                len: self.len(),
                remaining_blocks: [].iter(),
            },
        }
    }

    /// Computes how many bits would be set in the union between two bitsets.
    ///
    /// This is potentially much faster than using `union(other).count()`. Unlike
    /// other methods like using [`union_with`] followed by [`count_ones`], this
    /// does not mutate in place or require separate allocations.
    #[inline]
    pub fn union_count(self, other: FixedBitCell) -> usize {
        let me = self.as_slice();
        let other = other.as_slice();
        let count = Self::batch_count_ones(me.iter().zip(other.iter()).map(|(x, y)| *x | *y));
        match other.len().cmp(&me.len()) {
            Ordering::Greater => count + Self::batch_count_ones(other[me.len()..].iter().copied()),
            Ordering::Less => count + Self::batch_count_ones(me[other.len()..].iter().copied()),
            Ordering::Equal => count,
        }
    }

    /// Computes how many bits would be set in the intersection between two bitsets.
    ///
    /// This is potentially much faster than using `intersection(other).count()`. Unlike
    /// other methods like using [`intersect_with`] followed by [`count_ones`], this
    /// does not mutate in place or require separate allocations.
    #[inline]
    pub fn intersection_count(&self, other: FixedBitCell) -> usize {
        Self::batch_count_ones(
            self.as_slice()
                .iter()
                .zip(other.as_slice())
                .map(|(x, y)| *x & *y),
        )
    }

    /// Computes how many bits would be set in the difference between two bitsets.
    ///
    /// This is potentially much faster than using `difference(other).count()`. Unlike
    /// other methods like using [`difference_with`] followed by [`count_ones`], this
    /// does not mutate in place or require separate allocations.
    #[inline]
    pub fn difference_count(&self, other: FixedBitCell) -> usize {
        Self::batch_count_ones(
            self.as_slice()
                .iter()
                .zip(other.as_slice().iter())
                .map(|(x, y)| *x & !*y),
        ) + Self::batch_count_ones(self.as_slice().iter().skip(other.as_slice().len()).copied())
    }

    /// Computes how many bits would be set in the symmetric difference between two bitsets.
    ///
    /// This is potentially much faster than using `symmetric_difference(other).count()`. Unlike
    /// other methods like using [`symmetric_difference_with`] followed by [`count_ones`], this
    /// does not mutate in place or require separate allocations.
    #[inline]
    pub fn symmetric_difference_count(&self, other: FixedBitCell) -> usize {
        let me = self.as_slice();
        let other = other.as_slice();
        let count = Self::batch_count_ones(me.iter().zip(other.iter()).map(|(x, y)| *x ^ *y));
        match other.len().cmp(&me.len()) {
            Ordering::Greater => count + Self::batch_count_ones(other[me.len()..].iter().copied()),
            Ordering::Less => count + Self::batch_count_ones(me[other.len()..].iter().copied()),
            Ordering::Equal => count,
        }
    }

    /// Returns a lazy iterator over the intersection of two `FixedBitSet`s
    pub fn intersection(self, other: FixedBitCell<'a>) -> Intersection<'a> {
        Intersection {
            iter: self.ones(),
            other,
        }
    }

    /// Returns a lazy iterator over the union of two `FixedBitSet`s.
    pub fn union(self, other: FixedBitCell<'a>) -> Union<'a> {
        Union {
            iter: self.ones().chain(other.difference(self)),
        }
    }

    /// Returns a lazy iterator over the difference of two `FixedBitSet`s. The difference of `a`
    /// and `b` is the elements of `a` which are not in `b`.
    pub fn difference(self, other: FixedBitCell<'a>) -> Difference<'a> {
        Difference {
            iter: self.ones(),
            other,
        }
    }

    /// Returns a lazy iterator over the symmetric difference of two `FixedBitSet`s.
    /// The symmetric difference of `a` and `b` is the elements of one, but not both, sets.
    pub fn symmetric_difference(self, other: FixedBitCell<'a>) -> SymmetricDifference<'a> {
        SymmetricDifference {
            iter: self.difference(other).chain(other.difference(self)),
        }
    }

    /// Returns `true` if `self` has no elements in common with `other`. This
    /// is equivalent to checking for an empty intersection.
    pub fn is_disjoint(self, other: FixedBitCell) -> bool {
        self.as_simd_slice()
            .iter()
            .zip(other.as_simd_slice())
            .all(|(x, y)| (*x & *y).is_empty())
    }

    /// Returns `true` if the set is a subset of another, i.e. `other` contains
    /// at least all the values in `self`.
    pub fn is_subset(self, other: FixedBitCell) -> bool {
        let me = self.as_simd_slice();
        let other = other.as_simd_slice();
        me.iter()
            .zip(other.iter())
            .all(|(x, y)| x.andnot(*y).is_empty())
            && me.iter().skip(other.len()).all(|x| x.is_empty())
    }

    /// Returns `true` if the set is a superset of another, i.e. `self` contains
    /// at least all the values in `other`.
    pub fn is_superset(self, other: FixedBitCell) -> bool {
        other.is_subset(self)
    }

    /// # Safety
    /// Caller must ensures subblock is inside the range of blocks.len()
    #[inline]
    unsafe fn get_unchecked_mut(self, subblock: usize) -> &'a mut Block {
        // SAFETY: caller ensures the safety of this
        unsafe { &mut *self.data.as_ptr().cast::<Block>().add(subblock) }
    }

    /// Sets a bit to the provided `enabled` value.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn set(self, bit: usize, enabled: bool) {
        assert!(
            bit < self.bits,
            "set at index {} exceeds fixedbitset size {}",
            bit,
            self.bits
        );
        // SAFETY: The above assertion ensures that the block is inside the Vec's allocation.
        unsafe {
            self.set_unchecked(bit, enabled);
        }
    }

    /// Sets a bit to the provided `enabled` value without doing any bounds checking.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn set_unchecked(self, bit: usize, enabled: bool) {
        let (block, i) = div_rem(bit, Block::BITS as usize);
        // SAFETY: The above assertion ensures that the block is inside the Vec's allocation.
        let elt = unsafe { self.get_unchecked_mut(block) };
        if enabled {
            *elt |= 1 << i;
        } else {
            *elt &= !(1 << i);
        }
    }

    /// Enable `bit`.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn insert(self, bit: usize) {
        assert!(
            bit < self.bits,
            "insert at index {} exceeds fixedbitset size {}",
            bit,
            self.bits
        );
        // SAFETY: The above assertion ensures that the block is inside the Vec's allocation.
        unsafe {
            self.insert_unchecked(bit);
        }
    }

    /// Enable `bit` without any length checks.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn insert_unchecked(self, bit: usize) {
        let (block, i) = div_rem(bit, Block::BITS as usize);
        // SAFETY: The above assertion ensures that the block is inside the Vec's allocation.
        unsafe {
            *self.get_unchecked_mut(block) |= 1 << i;
        }
    }

    /// Disable `bit`.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn remove(self, bit: usize) {
        assert!(
            bit < self.bits,
            "remove at index {} exceeds fixedbitset size {}",
            bit,
            self.bits
        );
        // SAFETY: The above assertion ensures that the block is inside the Vec's allocation.
        unsafe {
            self.remove_unchecked(bit);
        }
    }

    /// Disable `bit` without any bounds checking.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn remove_unchecked(self, bit: usize) {
        let (block, i) = div_rem(bit, Block::BITS as usize);
        // SAFETY: The above assertion ensures that the block is inside the Vec's allocation.
        unsafe {
            *self.get_unchecked_mut(block) &= !(1 << i);
        }
    }

    /// Enable `bit`, and return its previous value.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn put(self, bit: usize) -> bool {
        assert!(
            bit < self.bits,
            "put at index {} exceeds fixedbitset size {}",
            bit,
            self.bits
        );
        // SAFETY: The above assertion ensures that the block is inside the Vec's allocation.
        unsafe { self.put_unchecked(bit) }
    }

    /// Enable `bit`, and return its previous value without doing any bounds checking.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn put_unchecked(self, bit: usize) -> bool {
        let (block, i) = div_rem(bit, Block::BITS as usize);
        // SAFETY: The above assertion ensures that the block is inside the Vec's allocation.
        unsafe {
            let word = self.get_unchecked_mut(block);
            let prev = *word & (1 << i) != 0;
            *word |= 1 << i;
            prev
        }
    }

    /// Toggle `bit` (inverting its state).
    ///
    /// ***Panics*** if **bit** is out of bounds
    #[inline]
    pub fn toggle(self, bit: usize) {
        assert!(
            bit < self.bits,
            "toggle at index {} exceeds fixedbitset size {}",
            bit,
            self.bits
        );
        // SAFETY: The above assertion ensures that the block is inside the Vec's allocation.
        unsafe {
            self.toggle_unchecked(bit);
        }
    }

    /// Toggle `bit` (inverting its state) without any bounds checking.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn toggle_unchecked(self, bit: usize) {
        let (block, i) = div_rem(bit, Block::BITS as usize);
        // SAFETY: The above assertion ensures that the block is inside the Vec's allocation.
        unsafe {
            *self.get_unchecked_mut(block) ^= 1 << i;
        }
    }

    /// Copies boolean value from specified bit to the specified bit.
    ///
    /// If `from` is out-of-bounds, `to` will be unset.
    ///
    /// **Panics** if **to** is out of bounds.
    #[inline]
    pub fn copy_bit(self, from: usize, to: usize) {
        assert!(
            to < self.bits,
            "copy to index {} exceeds fixedbitset size {}",
            to,
            self.bits
        );
        let enabled = self.contains(from);
        // SAFETY: The above assertion ensures that the block is inside the Vec's allocation.
        unsafe { self.set_unchecked(to, enabled) };
    }

    /// Copies boolean value from specified bit to the specified bit.
    ///
    /// Note: unlike `copy_bit`, calling this with an invalid `from`
    /// is undefined behavior.
    ///
    /// # Safety
    /// `to` must both be less than `self.len()`
    #[inline]
    pub unsafe fn copy_bit_unchecked(self, from: usize, to: usize) {
        // SAFETY: Caller must ensure that `from` is within bounds.
        let enabled = unsafe { self.contains_unchecked(from) };
        // SAFETY: Caller must ensure that `to` is within bounds.
        unsafe { self.set_unchecked(to, enabled) };
    }
}

impl<'a> Binary for FixedBitCell<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        if f.alternate() {
            f.write_str("0b")?;
        }

        for i in 0..self.bits {
            if self.contains(i) {
                f.write_char('1')?;
            } else {
                f.write_char('0')?;
            }
        }

        Ok(())
    }
}

impl<'a> Display for FixedBitCell<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        Binary::fmt(&self, f)
    }
}

#[derive(Debug, Clone)]
pub struct FixedBitSlice<'a>(FixedBitCell<'a>);

// SAFETY: FixedBitSlice does not contains thread-local state
unsafe impl<'a> Send for FixedBitSlice<'a> {}

// SAFETY: No mutable access is allowed in this
unsafe impl<'a> Sync for FixedBitSlice<'a> {}

impl<'a> FixedBitSlice<'a> {
    /// # Safety
    /// - data must be a valid ptr.
    /// - blocks must be a valid range in data
    pub unsafe fn from_raw_parts(data: *const Block, bits: usize) -> FixedBitSlice<'a> {
        FixedBitSlice(FixedBitCell::from_raw_parts(data, bits))
    }

    /// View the bitset as a slice of `Block` blocks
    #[inline]
    pub fn as_slice(&self) -> &'a [Block] {
        self.0.as_slice()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn is_clear(&self) -> bool {
        self.0.is_clear()
    }

    /// Finds the lowest set bit in the bitset.
    ///
    /// Returns `None` if there aren't any set bits.
    #[inline]
    pub fn minimum(&self) -> Option<usize> {
        self.0.minimum()
    }

    /// Finds the highest set bit in the bitset.
    ///
    /// Returns `None` if there aren't any set bits.
    #[inline]
    pub fn maximum(&self) -> Option<usize> {
        self.0.maximum()
    }

    /// `true` if all bits in the [`FixedBitSet`] are set.
    ///
    /// This is equivalent to [`bitset.count_ones(..) == bitset.len()`](FixedBitSet::count_ones).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.0.is_full()
    }

    /// Return **true** if the bit is enabled in the **FixedBitSet**,
    /// **false** otherwise.
    ///
    /// Note: bits outside the capacity are always disabled.
    ///
    /// Note: Also available with index syntax: `bitset[bit]`.
    #[inline]
    pub fn contains(&self, bit: usize) -> bool {
        self.0.contains(bit)
    }

    /// Return **true** if the bit is enabled in the **FixedBitSet**,
    /// **false** otherwise.
    ///
    /// Note: unlike `contains`, calling this with an invalid `bit`
    /// is undefined behavior.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn contains_unchecked(&self, bit: usize) -> bool {
        self.0.contains_unchecked(bit)
    }

    /// Checks if the bitset contains every bit in the given range.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn contains_all_in_range<T: IndexRange>(&self, range: T) -> bool {
        self.0.contains_all_in_range(range)
    }

    /// Checks if the bitset contains at least one set bit in the given range.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn contains_any_in_range<T: IndexRange>(&self, range: T) -> bool {
        self.0.contains_any_in_range(range)
    }

    /// Count the number of set bits in the given bit range.
    ///
    /// This function is potentially much faster than using `ones(other).count()`.
    /// Use `..` to count the whole content of the bitset.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn count_ones<T: IndexRange>(&self, range: T) -> usize {
        self.0.count_ones(range)
    }

    /// Count the number of unset bits in the given bit range.
    ///
    /// This function is potentially much faster than using `zeroes(other).count()`.
    /// Use `..` to count the whole content of the bitset.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn count_zeroes<T: IndexRange>(&self, range: T) -> usize {
        self.0.count_zeroes(range)
    }

    /// Iterates over all enabled bits.
    ///
    /// Iterator element is the index of the `1` bit, type `usize`.
    #[inline]
    pub fn ones(&self) -> Ones<'_> {
        self.0.ones()
    }

    /// Iterates over all disabled bits.
    ///
    /// Iterator element is the index of the `0` bit, type `usize`.
    #[inline]
    pub fn zeroes(&self) -> Zeroes<'_> {
        self.0.zeroes()
    }

    /// Computes how many bits would be set in the union between two bitsets.
    ///
    /// This is potentially much faster than using `union(other).count()`. Unlike
    /// other methods like using [`union_with`] followed by [`count_ones`], this
    /// does not mutate in place or require separate allocations.
    #[inline]
    pub fn union_count(&self, other: &FixedBitSlice) -> usize {
        self.0.union_count(other.0)
    }

    /// Computes how many bits would be set in the intersection between two bitsets.
    ///
    /// This is potentially much faster than using `intersection(other).count()`. Unlike
    /// other methods like using [`intersect_with`] followed by [`count_ones`], this
    /// does not mutate in place or require separate allocations.
    #[inline]
    pub fn intersection_count(&self, other: &FixedBitSlice) -> usize {
        self.0.intersection_count(other.0)
    }

    /// Computes how many bits would be set in the difference between two bitsets.
    ///
    /// This is potentially much faster than using `difference(other).count()`. Unlike
    /// other methods like using [`difference_with`] followed by [`count_ones`], this
    /// does not mutate in place or require separate allocations.
    #[inline]
    pub fn difference_count(&self, other: &FixedBitSlice) -> usize {
        self.0.difference_count(other.0)
    }

    /// Computes how many bits would be set in the symmetric difference between two bitsets.
    ///
    /// This is potentially much faster than using `symmetric_difference(other).count()`. Unlike
    /// other methods like using [`symmetric_difference_with`] followed by [`count_ones`], this
    /// does not mutate in place or require separate allocations.
    #[inline]
    pub fn symmetric_difference_count(&self, other: &FixedBitSlice) -> usize {
        self.0.symmetric_difference_count(other.0)
    }

    /// Returns a lazy iterator over the intersection of two `FixedBitSet`s
    pub fn intersection(&'a self, other: &'a FixedBitSlice<'a>) -> Intersection<'a> {
        self.0.intersection(other.0)
    }

    /// Returns a lazy iterator over the union of two `FixedBitSet`s.
    pub fn union(&'a self, other: &'a FixedBitSlice<'a>) -> Union<'a> {
        self.0.union(other.0)
    }

    /// Returns a lazy iterator over the difference of two `FixedBitSet`s. The difference of `a`
    /// and `b` is the elements of `a` which are not in `b`.
    pub fn difference(&'a self, other: &'a FixedBitSlice<'a>) -> Difference<'a> {
        self.0.difference(other.0)
    }

    /// Returns a lazy iterator over the symmetric difference of two `FixedBitSet`s.
    /// The symmetric difference of `a` and `b` is the elements of one, but not both, sets.
    pub fn symmetric_difference(&'a self, other: &'a FixedBitSlice<'a>) -> SymmetricDifference<'a> {
        self.0.symmetric_difference(other.0)
    }

    /// Returns `true` if `self` has no elements in common with `other`. This
    /// is equivalent to checking for an empty intersection.
    pub fn is_disjoint(&self, other: &FixedBitSlice) -> bool {
        self.0.is_disjoint(other.0)
    }

    /// Returns `true` if the set is a subset of another, i.e. `other` contains
    /// at least all the values in `self`.
    pub fn is_subset(&self, other: &FixedBitSlice) -> bool {
        self.0.is_subset(other.0)
    }

    /// Returns `true` if the set is a superset of another, i.e. `self` contains
    /// at least all the values in `other`.
    pub fn is_superset(&self, other: &FixedBitSlice) -> bool {
        other.is_subset(self)
    }
}

#[derive(Debug)]
pub struct FixedBitSliceMut<'a>(FixedBitCell<'a>);

// SAFETY: FixedBitSliceMut does not contains thread-local state
unsafe impl<'a> Send for FixedBitSliceMut<'a> {}

// SAFETY: No cross thread mutable access is allowed
unsafe impl<'a> Sync for FixedBitSliceMut<'a> {}

impl<'a> FixedBitSliceMut<'a> {
    /// # Safety
    /// - data must be a valid ptr.
    /// - blocks must be a valid range in data
    pub unsafe fn from_raw_parts(data: *mut Block, bits: usize) -> FixedBitSliceMut<'a> {
        FixedBitSliceMut(FixedBitCell::from_raw_parts(data, bits))
    }

    pub fn as_readonly(&self) -> FixedBitSlice<'_> {
        FixedBitSlice(self.0)
    }

    /// Sets a bit to the provided `enabled` value.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn set(&mut self, bit: usize, enabled: bool) {
        self.0.set(bit, enabled);
    }

    /// Sets a bit to the provided `enabled` value without doing any bounds checking.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn set_unchecked(&mut self, bit: usize, enabled: bool) {
        self.0.set_unchecked(bit, enabled);
    }

    /// Enable `bit`.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn insert(&mut self, bit: usize) {
        self.0.insert(bit);
    }

    /// Enable `bit` without any length checks.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn insert_unchecked(&mut self, bit: usize) {
        self.0.insert_unchecked(bit);
    }

    /// Disable `bit`.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn remove(&mut self, bit: usize) {
        self.0.remove(bit);
    }

    /// Disable `bit` without any bounds checking.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn remove_unchecked(&mut self, bit: usize) {
        self.0.remove_unchecked(bit);
    }

    /// Enable `bit`, and return its previous value.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn put(&mut self, bit: usize) -> bool {
        self.0.put(bit)
    }

    /// Enable `bit`, and return its previous value without doing any bounds checking.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn put_unchecked(&mut self, bit: usize) -> bool {
        self.0.put_unchecked(bit)
    }

    /// Toggle `bit` (inverting its state).
    ///
    /// ***Panics*** if **bit** is out of bounds
    #[inline]
    pub fn toggle(&mut self, bit: usize) {
        self.0.toggle(bit)
    }

    /// Toggle `bit` (inverting its state) without any bounds checking.
    ///
    /// # Safety
    /// `bit` must be less than `self.len()`
    #[inline]
    pub unsafe fn toggle_unchecked(&mut self, bit: usize) {
        self.0.toggle_unchecked(bit)
    }

    /// Copies boolean value from specified bit to the specified bit.
    ///
    /// If `from` is out-of-bounds, `to` will be unset.
    ///
    /// **Panics** if **to** is out of bounds.
    #[inline]
    pub fn copy_bit(&mut self, from: usize, to: usize) {
        self.0.copy_bit(from, to)
    }

    /// Copies boolean value from specified bit to the specified bit.
    ///
    /// Note: unlike `copy_bit`, calling this with an invalid `from`
    /// is undefined behavior.
    ///
    /// # Safety
    /// `to` must both be less than `self.len()`
    #[inline]
    pub unsafe fn copy_bit_unchecked(&mut self, from: usize, to: usize) {
        self.0.copy_bit_unchecked(from, to)
    }
}
