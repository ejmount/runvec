pub struct RunIterator<T, Iter>
where
    T: PartialEq,
    Iter: Iterator<Item = T>,
{
    iter: Iter,
    current: Option<T>,
    count: usize,
}

impl<T, Iter> RunIterator<T, Iter>
where
    T: PartialEq,
    Iter: Iterator<Item = T>,
{
    pub fn new(iter: Iter) -> RunIterator<T, Iter> {
        RunIterator {
            iter,
            current: None,
            count: 0,
        }
    }
}

impl<T, Iter> Iterator for RunIterator<T, Iter>
where
    T: PartialEq,
    Iter: Iterator<Item = T>,
{
    type Item = (T, usize);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let current_item = std::mem::replace(&mut self.current, None); // Seems to be issues with doing by-ref and by-val matching at once, so do it all by-val
            match (self.iter.next(), current_item) {
                (Some(item), None) => {
                    self.current = Some(item);
                    self.count = 1;
                }
                (next, Some(current)) => {
                    if next.is_some() && current.eq(next.as_ref().unwrap()) {
                        self.count += 1;
                        self.current = Some(current);
                    } else {
                        let out_group = (current, self.count);
                        self.count = 1;
                        self.current = next;
                        return Some(out_group);
                    }
                }
                (None, None) => {
                    // No need to replace self.current, it's already None
                    return None;
                }
            }
        }
    }
}

pub struct ExpandingIterator<T, Iter>
where
    T: Clone,
    Iter: Iterator<Item = (T, usize)>,
{
    iter: Iter,
    cur_item: Option<(T, usize)>,
}

impl<T, Iter> ExpandingIterator<T, Iter>
where
    T: Clone,
    Iter: Iterator<Item = (T, usize)>,
{
    pub fn new(mut iter: Iter) -> ExpandingIterator<T, Iter> {
        ExpandingIterator {
            cur_item: iter.next(),
            iter,
        }
    }
}

impl<T, Iter> Iterator for ExpandingIterator<T, Iter>
where
    T: Clone,
    Iter: Iterator<Item = (T, usize)>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.cur_item {
            &mut Some((ref element, ref mut count)) => {
                *count -= 1;
                let e = element.clone();
                if *count == 0 {
                    self.cur_item = self.iter.next();
                }
                return Some(e);
            }
            &mut None => {
                self.cur_item = self.iter.next();
                return None;
            }
        }
    }
}
