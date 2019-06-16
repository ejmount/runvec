use crate::iter::{ExpandingIterator, RunLengthIterator};

/// A marker trait indicating which types can be usefully used in [RunLenVec]. Automatically implemented for all compatible types.
pub trait RunLenCompressible: Clone + PartialEq {}
impl<T> RunLenCompressible for T where T: Clone + PartialEq {}


#[derive(Clone)]
pub struct RunLenVec<T: RunLenCompressible> {
    inner: Vec<(T, usize)>,
    total_size: usize,
}

impl<T: RunLenCompressible> RunLenVec<T> {
    // Returns the index of the segment containing the given expanded index, along with the offset between the start of that segment and the start of the index
    fn segment_containing_index(&self, index: usize) -> Option<(usize, usize)> {
        if index < self.total_size {
            let mut total_span = 0;
            let mut current_ind = 0;
            while (total_span + self.inner[current_ind].1) <= index {
                total_span += self.inner[current_ind].1;
                current_ind += 1;
            }
            return Some((current_ind, index - total_span));
        } else {
            None
        }
    }
    fn compact(&mut self) {
        let inner = &mut self.inner;
        for index in (1..inner.len()).rev() {
            if inner[index].0 == inner[index - 1].0 {
                inner[index - 1].1 += inner[index].1;
                inner[index].1 = 0;
            }
        }
        self.inner.retain(|&(_, c)| c > 0);
        // self.total_size has not changed.
    }

    fn update_size(&mut self) {
        self.total_size = self.inner.iter().map(|(_, s)| s).sum();
    }

    /// Create a empty vec.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// let rlv = RunLenVec::<u32>::new();
    /// assert!(rlv.is_empty());
    /// ```
    pub fn new() -> RunLenVec<T> {
        RunLenVec {
            inner: vec![],
            total_size: 0,
        }
    }


    /// Constructs a new vector with the given capacity. See [`capacity()`][capacity] for how this differs from the capacity of a `Vec`.
    /// 
    /// [capacity]: #method.capacity
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// let mut rlv = RunLenVec::with_capacity(1);
    /// for _ in 0..1_000_000 {
    ///     rlv.push(1);
    /// }
    /// # assert_eq!(rlv.compressed_len(), 1);
    /// assert_eq!(rlv.capacity(), 1);
    /// assert_eq!(rlv.len(), 1_000_000);
    /// ```
    pub fn with_capacity(capacity: usize) -> RunLenVec<T> {
        RunLenVec {
            inner: Vec::with_capacity(capacity),
            total_size: 0,
        }
    }

    /// Returns the capacity of the underlying storage. The vector can store this many _non-contigious runs_ before having to resize - it is not a limit on the total number of elements.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// let mut rlv = RunLenVec::from_raw(vec![(1,1000000)]);
    /// # assert_eq!(rlv.compressed_len(), 1);
    /// assert_eq!(rlv.len(), 1000000);
    /// assert!(rlv.capacity() < 1000000);
    /// ```
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Increases the capacity to hold at least `additional` more runs of any size, with the same disclaimers as [`Vec::reserve`][reserve]
    /// 
    /// [reserve]: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.reserve
    /// # Panics
    /// Panics if the new capacity overflows `usize`
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let mut rlv = RunLenVec::from_iter(vec![1]);
    /// rlv.reserve(10);
    /// assert!(rlv.capacity() >= 11);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    /// Shrinks the vector as much as possible, as per [`Vec::shrink_to_fit`][shrink]
    /// 
    /// [shrink]: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.shrink_to_fit
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    /// Shortens the vector by dropping elements until it has the given uncompressed length. Has no effect if the vector is already short enough.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let mut rlv = RunLenVec::from_iter(vec![1,1,1,1,2,1,1,1]);
    /// rlv.truncate(3);
    /// assert_eq!(rlv.len(), 3);
    /// assert_eq!(rlv.compressed_len(), 1);
    /// ```
    pub fn truncate(&mut self, len: usize) {
        if len >= self.total_size {
            return;
        }
        let mut elements_to_remove = self.total_size - len;
        self.total_size -= elements_to_remove;
        while elements_to_remove > 0 {
            let last_group = self.inner.last_mut().unwrap(); // Inner should be non-empty if self.total_size > 0, implied by total_size > (a usize)
            let last_group_size = last_group.1;
            if last_group_size <= elements_to_remove {
                elements_to_remove -= last_group_size;
                self.inner.pop();
            } else {
                last_group.1 -= elements_to_remove;
                elements_to_remove = 0;
            }
        }
    }

    /// Insert the given element into the given logical index, splitting a run if required.
    /// # Panics 
    /// Panics if a single run contains more than `usize` elements. 
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let mut rlv = RunLenVec::from_iter(vec![1,1,1]);
    /// rlv.insert(2, 2);
    /// # assert_eq!(rlv.len(), 4);
    /// # assert_eq!(rlv.compressed_len(), 3);
    /// assert_eq!(rlv.to_vec(), vec![1,1,2,1]);
    /// ```
    pub fn insert(&mut self, index: usize, element: T) {
        if index == self.total_size {
            self.push(element);
        } else {
            let (segment_index, offset) = self.segment_containing_index(index).unwrap();
            if element == self.inner[segment_index].0 {
                self.inner[segment_index].1 += 1;
            } else {
                let (orig_element, size) = &mut self.inner[segment_index];
                let excess = *size - offset;
                *size = offset;
                let new_element = orig_element.clone();
                self.inner.insert(segment_index + 1, (element, 1));
                if excess > 0 {
                    self.inner.insert(segment_index + 2, (new_element, excess));
                }
            }
            self.total_size += 1;
        }
    }

    /// Removes the element at a given index.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let mut rlv = RunLenVec::from_iter(vec![1,1,2,1,1]);
    /// rlv.remove(2);
    /// # assert_eq!(rlv.len(), 4);
    /// # assert_eq!(rlv.compressed_len(), 1);
    /// assert_eq!( rlv.to_vec(), vec![1,1,1,1]);
    /// ```
    pub fn remove(&mut self, index: usize) -> T {
        let (segment_index, _) = self.segment_containing_index(index).unwrap();
        let segment = &mut self.inner[segment_index];
        segment.1 -= 1;
        let element = segment.0.clone();
        if segment.1 == 0 {
            self.inner.remove(segment_index);
        }
        self.total_size -= 1;
        return element;
    }

    /// Applies the given closure to each element and keeps those elements where the closure returns true, as per the same method on [`Vec`][retain], except the closure is only called once for each run.
    ///
    /// [retain]: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.retain
    ///
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let mut items = RunLenVec::from_iter(vec![1,1,1,2,2,2,3,3,3]);
    /// let mut calls = 0;
    /// items.retain(|x: &u32| {calls += 1; *x % 2 != 0});
    /// assert_eq!(calls, 3);
    /// assert_eq!(items.len(), 6);
    /// ```
    pub fn retain<F>(&mut self, mut func: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.inner.retain(|(e, _)| func(e));
        self.update_size();
    }
    /// Pushes a new element to the rightmost end.
    /// # Panics
    /// Panics if a single run contains more than `usize` elements. 
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// let mut rlv = RunLenVec::new();
    /// rlv.push(0);
    /// rlv.push(0);
    /// rlv.push(1);
    /// # assert_eq!(rlv.len(), 3);
    /// # assert_eq!(rlv.compressed_len(), 2);
    /// assert_eq!(rlv.to_vec(), vec![0,0,1]);
    /// ```
    pub fn push(&mut self, element: T) {
        self.total_size += 1;
        if let Some(ref mut l) = self.inner.last_mut() {
            if l.0 == element {
                l.1 += 1;
                return;
            }
        }
        self.inner.push((element, 1))
    }

    /// Removes and returns the rightmost element.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let mut rlv = RunLenVec::from_iter(vec![2,1,1,1]);
    /// assert_eq!(rlv.pop(), Some(1));
    /// assert_eq!(rlv.pop(), Some(1));
    /// # assert_eq!(rlv.len(), 2);
    /// assert_eq!(rlv.pop(), Some(1));
    /// assert_eq!(rlv.pop(), Some(2));
    /// assert_eq!(rlv.pop(), None);
    /// # assert_eq!(rlv.len(), 0);
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        if let Some(last) = self.inner.last_mut() {
            self.total_size -= 1;
            last.1 -= 1;
            let element = if last.1 == 0 {
                self.inner.pop().unwrap().0
            } else {
                last.0.clone()
            };
            return Some(element);
        } else {
            return None;
        }
    }


    /// Drops all elements.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let mut rlv = RunLenVec::from_iter(vec![1,2,3,4,5]);
    /// rlv.clear();
    /// assert_eq!(rlv.len(), 0);
    /// assert_eq!(rlv.compressed_len(), 0);
    /// ```
    pub fn clear(&mut self) {
        self.inner.clear();
        self.total_size = 0;
    }

    /// Returns the total number of elements. Potentially larger than [`capacity`](capacity) when adjacent elements are equal.
    /// [capacity]: #method.capacity
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let rlv = RunLenVec::from_iter(vec![1,1,1,1,1]);
    /// assert_eq!(rlv.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.total_size
    }

    /// Returns the number of distinct runs of non-equal elements. This value is limited by the capacity.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let rlv = RunLenVec::from_iter(vec![1,1,1,1,1]);
    /// assert_eq!(rlv.compressed_len(), 1);
    /// ```
    pub fn compressed_len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` when there are no elements.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// assert!(!RunLenVec::from_iter(vec![1]).is_empty());
    /// assert!(RunLenVec::from_iter(vec![0u32; 0]).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }


    /// Splits the collection at the given index, leaving the elements `[0, at)` in the given instance and returning a new instance containing the elements between `[at, len)`.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let mut orig_rlv = RunLenVec::from_iter(vec![1,1,1,2,2]);
    /// let splitee = orig_rlv.split_off(2);
    /// assert_eq!(orig_rlv.to_vec(), vec![1,1]);
    /// assert_eq!(splitee.to_vec(), vec![1,2,2]);
    /// ```
    pub fn split_off(&mut self, at: usize) -> RunLenVec<T> {
        let (segment, offset) = self.segment_containing_index(at).unwrap();
        let excess = self.inner[segment].1 - offset;
        self.inner[segment].1 -= excess;
        let mut new_vec = RunLenVec::with_capacity(self.inner.len() - at + 1);
        if excess > 0 {
            new_vec.inner.push((self.inner[segment].0.clone(), excess));
        }
        new_vec.inner.extend(self.inner.drain(segment + 1..));
        if self.inner[segment].1 == 0 {
            self.inner.remove(segment);
        }
        self.update_size();
        new_vec.update_size();
        return new_vec;
    }

    /// Resizes the collection so that `len` is equal to `new_len`, using the provided function to generate new elements if required.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let mut rlv = RunLenVec::from_iter(vec![1,1,1]);
    /// rlv.resize_with(5, || 0);
    /// assert_eq!(rlv.to_vec(), vec![1,1,1,0,0]);
    /// rlv.resize_with(2, || 0);
    /// assert_eq!(rlv.to_vec(), vec![1,1]);
    /// ```
    pub fn resize_with<F>(&mut self, new_len: usize, mut f: F)
    where
        F: FnMut() -> T,
    {
        if self.total_size > new_len {
            self.truncate(new_len)
        } else if self.total_size < new_len {
            self.inner.reserve(new_len - self.len());
            while self.total_size < new_len {
                self.push(f())
            }
        }
    }

    /// Resizes the collection so that `len` is equal to `new_len`, cloning the provided element as required if the collection needs to be larger.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let mut rlv = RunLenVec::from_iter(vec![1,1,1]);
    /// rlv.resize(5, 7);
    /// assert_eq!(rlv.to_vec(), vec![1,1,1,7,7]);
    /// rlv.resize(2, 7);
    /// assert_eq!(rlv.to_vec(), vec![1,1]);
    /// ```
    pub fn resize(&mut self, new_len: usize, value: T) {
        if new_len < self.total_size {
            self.truncate(new_len)
        } else {
            match self.inner.last_mut() {
                Some(ref mut last) if last.0 == value => {
                    (*last).1 += new_len - self.total_size;
                    self.total_size = new_len;
                }
                _ => {
                    self.inner.push((value, new_len - self.total_size));
                    self.total_size = new_len;
                }
            }
        }
    }

    /// Returns a reference to the first element. Returns `None` if the collection is empty.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// assert_eq!(*RunLenVec::from_iter(vec![1]).first().unwrap(), 1);
    /// assert_eq!(RunLenVec::from_iter(vec![0u32; 0]).first(), None);
    /// ```
    pub fn first(&self) -> Option<&T> {
        self.inner.first().map(|(ref e, _)| e)
    }

    /// Returns a reference to the last element. Returns `None` if the collection is empty.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// assert_eq!(*RunLenVec::from_iter(vec![1,2,3]).last().unwrap(), 3);
    /// assert_eq!(RunLenVec::from_iter(vec![0u32; 0]).last(), None);
    /// ```
    pub fn last(&self) -> Option<&T> {
        self.inner.last().as_ref().map(|(ref e, _)| e)
    }

    /// Returns a reference to the element at the given index, if within range.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let rlv = RunLenVec::from_iter(vec![1,1,1,2,3]); 
    /// assert_eq!(*rlv.get(0).unwrap(), 1);
    /// assert_eq!(*rlv.get(4).unwrap(), 3);
    /// assert_eq!(rlv.get(5), None);
    /// ```
    pub fn get(&self, index: usize) -> Option<&T> {
        self.segment_containing_index(index)
            .and_then(|(i, _)| self.inner.get(i))
            .map(|(ref e, _)| e)
    }
    /// Reverses the collection in place.
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let mut rlv = RunLenVec::from_iter(vec![1,1,2,2,3]); 
    /// rlv.reverse();
    /// assert_eq!(rlv.to_vec(), vec![3,2,2,1,1]);
    /// ```
    pub fn reverse(&mut self) {
        self.inner.reverse()
    }

    /// Returns an iterator over the collection
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let x = RunLenVec::from_iter(vec![1, 2, 4]);
    /// let mut iterator = x.iter();
    /// assert_eq!(iterator.next(), Some(&1));
    /// assert_eq!(iterator.next(), Some(&2));
    /// assert_eq!(iterator.next(), Some(&4));
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<T> {
        Iter::new(&self.inner)
    }

    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.sort_by(T::cmp)
    }
    pub fn sort_by<F>(&mut self, mut f: F)
    where
        F: FnMut(&T, &T) -> std::cmp::Ordering,
    {
        self.inner.sort_by(|(a, _), (b, _)| f(a, b));
        self.compact();
    }
    pub fn sort_by_key<F, K>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.inner.sort_by_key(|(e, _)| f(e));
        self.compact();
    }
    pub fn sort_by_cached_key<F, K>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.inner.sort_by_cached_key(|(e, _)| f(e));
        self.compact();
    }

    /// Copies and expands `self` into a new Vec
    /// ```
    /// # use crate::runvec::RunLenVec;
    /// # use std::iter::FromIterator;
    /// let exp = vec![1,1,2,3,3];
    /// let rlv = RunLenVec::from_iter(exp.clone());
    /// assert_eq!(rlv.to_vec(), exp);
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().map(Clone::clone).collect()
    }
}

impl<T> std::hash::Hash for RunLenVec<T>
where
    T: std::hash::Hash + RunLenCompressible,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let hash_item = |e: &T| e.hash(state);
        self.iter().for_each(hash_item);
    }
}

impl<T> std::cmp::PartialEq for RunLenVec<T>
where
    T: RunLenCompressible,
{
    fn eq(&self, other: &Self) -> bool {
        if self.total_size != other.total_size || self.inner.len() != other.inner.len() {
            return false;
        } else {
            Iterator::zip(self.inner.iter(), other.inner.iter())
                .map(|(x, y)| x == y)
                .all(|b| b)
        }
    }
}
impl<T> std::cmp::Eq for RunLenVec<T> where T: RunLenCompressible + Eq {}

impl<T> std::cmp::PartialOrd for RunLenVec<T>
where
    T: PartialOrd + RunLenCompressible,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.total_size.cmp(&other.total_size) {
            std::cmp::Ordering::Equal => self.iter().partial_cmp(other.iter()),
            order => Some(order),
        }
    }
}
impl<T> std::cmp::Ord for RunLenVec<T>
where
    T: Ord + RunLenCompressible,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<T> std::fmt::Debug for RunLenVec<T>
where
    T: RunLenCompressible + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "RunLenVec {{ {} items: ", self.total_size)?;
        for (element, count) in &self.inner {
            write!(f, "({:?}, {:?}), ", count, element)?
        }
        write!(f, "}}")
    }
}

impl<T: RunLenCompressible> std::iter::FromIterator<T> for RunLenVec<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let elements: Vec<_> = RunLengthIterator::new(iter.into_iter()).collect();
        RunLenVec {
            total_size: elements.iter().map(|&(_, c)| c).sum(),
            inner: elements,
        }
    }
}

impl<T: RunLenCompressible> Extend<T> for RunLenVec<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        let mut new_elements = RunLengthIterator::new(iter.into_iter()).peekable();
        let first_group = new_elements.peek();
        if let Some(group) = first_group {
            if let Some(ref mut last) = self.inner.last_mut() {
                if last.0 == group.0 {
                    last.1 += group.1;
                    new_elements.next(); // Throw away the now-redundant duplicate group
                }
            }
        }
        self.inner.extend(new_elements);
    }
}

impl<'a, T> Extend<&'a T> for RunLenVec<T>
where
    T: RunLenCompressible,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a T>,
    {
        self.extend(iter.into_iter().map(Clone::clone))
    }
}

impl<T: RunLenCompressible> Default for RunLenVec<T> {
    fn default() -> Self {
        RunLenVec::new()
    }
}

impl<T: RunLenCompressible> std::ops::Index<usize> for RunLenVec<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        self.get(index).unwrap()
    }
}

impl<T: RunLenCompressible> IntoIterator for RunLenVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> IntoIter<T> {
        IntoIter(ExpandingIterator::new(self.inner.into_iter()))
    }
}

impl<'a, T: RunLenCompressible> IntoIterator for &'a RunLenVec<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[derive(Clone)]
struct MiniRefIter<'a, T: 'a>(&'a [(T, usize)], usize);
impl<'a, T> Iterator for MiniRefIter<'a, T> {
    type Item = (&'a T, usize);
    fn next(&mut self) -> Option<(&'a T, usize)> {
        let e = self.0.get(self.1).map(|&(ref e, c)| (e, c));
        self.1 += 1;
        return e;
    }
}

#[derive(Clone)]
pub struct Iter<'a, T: Clone>(ExpandingIterator<&'a T, MiniRefIter<'a, T>>);

impl<'a, T: Clone> Iter<'a, T> {
    fn new(v: &[(T, usize)]) -> Iter<T> {
        Iter(ExpandingIterator::new(MiniRefIter(v, 0)))
    }
}
impl<'a, T: Clone> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        self.0.next()
    }
}

#[derive(Clone)]
pub struct IntoIter<T: Clone>(ExpandingIterator<T, std::vec::IntoIter<(T, usize)>>);
impl<T: Clone> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.0.next()
    }
}

#[test]
fn compaction_test() {
    let mut rlv = RunLenVec {
        inner: vec![
            (1, 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 1),
            (1, 1),
            (1, 1),
            (1, 1),
        ],
        total_size: 10,
    };
    rlv.compact();
    assert_eq!(rlv.inner.len(), 3);
    assert_eq!(rlv.len(), 10);
}
