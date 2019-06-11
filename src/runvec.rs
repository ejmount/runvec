use crate::iter::{ExpandingIterator, RunIterator};

pub trait RunLenCompressible: Clone + PartialEq {}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RunLenVec<T: RunLenCompressible> {
    inner: Vec<(T, usize)>,
    total_size: usize,
}

impl<T: RunLenCompressible> RunLenVec<T> {
    fn segment_containing_index(&self, index: usize) -> Option<(usize, usize)> {
        if index < self.total_size {
            let mut total_span = 0;
            let mut current_ind = 0;
            while (total_span + self.inner[current_ind].1) < index {
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
        for index in (inner.len() - 1)..0 {
            if inner[index].0 == inner[index - 1].0 {
                inner[index - 1].1 += inner[index].1;
                inner[index].1 = 0;
            }
        }
        self.inner.retain(|&(_, c)| c > 0);
        // self.total_size _should_ be unchanged
    }
    fn update_size(&mut self) {
        self.total_size = self.inner.iter().map(|(_, s)| s).sum();
    }

    pub fn new() -> RunLenVec<T> {
        RunLenVec {
            inner: vec![],
            total_size: 0,
        }
    }
    pub fn with_capacity(capacity: usize) -> RunLenVec<T> {
        RunLenVec {
            inner: Vec::with_capacity(capacity),
            total_size: 0,
        }
    }
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }
    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional)
    }
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }
    pub fn truncate(&mut self, len: usize) {
        if len >= self.total_size {
            return;
        }
        let mut elements_to_remove = self.total_size - len;
        while elements_to_remove > 0 {
            let last_group = self.inner.last_mut().unwrap(); // Inner should be non-empty if self.total_size > 0, implied by total_size > (a usize)
            let last_group_size = last_group.1;
            if last_group_size <= elements_to_remove {
                std::mem::drop(last_group);
                elements_to_remove -= last_group_size;
                self.inner.pop();
            } else {
                last_group.1 -= elements_to_remove;
            }
        }
        self.total_size = self.total_size - elements_to_remove;
    }

    pub fn insert(&mut self, index: usize, element: T) {
        let (segment_index, offset) = self.segment_containing_index(index).unwrap();
        if element == self.inner[segment_index].0 {
            self.inner[segment_index].1 += 1;
        } else {
            let (orig_element, size) = &mut self.inner[segment_index];
            let excess = *size - offset;
            *size -= offset;
            let new_element = orig_element.clone();
            self.inner.insert(segment_index + 1, (element, 1));
            self.inner.insert(segment_index + 2, (new_element, excess));
        }
        self.total_size += 1;
    }
    pub fn remove(&mut self, index: usize) -> T {
        let (segment_index, _) = self.segment_containing_index(index).unwrap();
        let segment = self.inner.get_mut(segment_index).unwrap();
        segment.1 -= 1;
        let element = segment.0.clone();
        if segment.1 == 0 {
            self.inner.remove(segment_index);
        }
        self.total_size -= 1;
        return element;
    }
    pub fn retain<F>(&mut self, mut func: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.inner.retain(|(e, _)| func(e));
        self.update_size();
    }
    pub fn push(&mut self, element: T) {
        if (!self.inner.is_empty()) && self.inner.last().unwrap().0 == element {
            self.inner.last_mut().unwrap().1 += 1;
        } else {
            self.inner.push((element, 1))
        }
        self.total_size += 1;
    }
    pub fn pop(&mut self) -> Option<T> {
        if let Some(last) = self.inner.last_mut() {
            last.1 -= 1;
            let element = last.0.clone();
            if last.1 == 0 {
                self.inner.pop();
            }
            return Some(element);
        } else {
            return None;
        }
    }

    pub fn append(&mut self, other: &mut Vec<T>) {
        self.extend(other.drain(..))
    }
    pub fn join(&mut self, other: &mut Vec<(T, usize)>) {
        self.inner.append(other)
    }
    pub fn clear(&mut self) {
        self.inner.clear()
    }
    pub fn len(&self) -> usize {
        self.total_size
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
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
    pub fn first(&self) -> Option<&T> {
        self.inner.first().map(|(ref e, _)| e)
    }
    pub fn last(&self) -> Option<&T> {
        self.inner.last().as_ref().map(|(ref e, _)| e)
    }
    pub fn get(&self, index: usize) -> Option<&T> {
        self.segment_containing_index(index)
            .and_then(|(i, _)| self.inner.get(i))
            .map(|(ref e, _)| e)
    }
    pub fn reverse(&mut self) {
        self.inner.reverse()
    }

    pub fn iter<'a>(&'a self) -> Iter<'a, T> {
        Iter::new(&self.inner)
    }
    pub fn into_iter(self) -> IntoIter<T> {
        IntoIter(ExpandingIterator::new(self.inner.into_iter()))
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
        let elements: Vec<_> = RunIterator::new(iter.into_iter()).collect();
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
        let mut new_elements = RunIterator::new(iter.into_iter()).peekable();
        let first_group = new_elements.peek();
        if let Some(group) = first_group {
            if let Some(last) = self.inner.last() {
                if last.0 == group.0 {
                    self.inner.last_mut().unwrap().1 += group.1;
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
        self.into_iter()
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
struct MiniRefIter<'a, T>(&'a Vec<(T, usize)>, usize);
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
    fn new(v: &'a Vec<(T, usize)>) -> Iter<'a, T> {
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
