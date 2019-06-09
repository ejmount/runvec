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

impl<T, I> Iterator for RunIterator<T, I>
where
    T: PartialEq,
    I: Iterator<Item = T>,
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
