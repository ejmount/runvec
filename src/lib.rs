mod iter;
mod runvec;

pub use crate::runvec::RunLenVec;
pub use iter::{ExpandingIterator, RunIterator};

#[cfg(test)]
mod tests {
    use crate::iter::*;
    #[test]
    fn runiter_test() {
        let expected = vec![(1, 3), (5, 1), (6, 1), (3, 1), (2, 3)];
        let test_items = vec![1, 1, 1, 5, 6, 3, 2, 2, 2];
        let ri = RunIterator::new(test_items.into_iter());
        assert_eq!(ri.collect::<Vec<_>>(), expected);
    }

    #[test]
    fn expandingiter_test() {
        let expected = vec![1, 1, 1, 2, 3, 3];
        let test_items = vec![(1, 3), (2, 1), (3, 2)];
        let ri = ExpandingIterator::new(test_items.into_iter());
        assert_eq!(ri.collect::<Vec<_>>(), expected);
    }

}
