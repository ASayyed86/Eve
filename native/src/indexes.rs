//-------------------------------------------------------------------------
// Indexes
//-------------------------------------------------------------------------

use std::collections::HashMap;
use ops::EstimateIter;
use ops::Change;
use std::cmp;
use std;
use std::collections::hash_map::Entry;

//-------------------------------------------------------------------------
// Utils
//-------------------------------------------------------------------------

pub fn ensure_len(vec:&mut Vec<i32>, len:usize) {
    if vec.len() < len {
        vec.resize(len, 0);
    }
}

pub fn get_delta(last:i32, next:i32) -> i32 {
    if last == 0 && next > 0 { 1 }
    else if last > 0 && next == 0 { -1 }
    else if last > 0 && next < 0 { -1 }
    else if last < 0 && next > 0 { 1 }
    else { 0 }
}

//-------------------------------------------------------------------------
// Level store
//-------------------------------------------------------------------------

#[derive(Debug)]
struct LevelStore {
    levels: Vec<u32>,
    ix: u32,
}

impl LevelStore {
    fn new() -> LevelStore {
        let mut store = LevelStore{ levels:vec![], ix:0 };
        store.add(0);
        store
    }

    pub fn fetch(&mut self, pos:u32, offset:u32) -> u32 {
        let actual = (pos + offset) as usize;
        // println!("  Fetching {:?}", actual);
        match self.levels[actual] {
            0 => self.add(actual),
            any => any
        }
    }

    pub fn get(&self, pos:u32, offset:u32) -> u32 {
        let actual = (pos + offset) as usize;
        self.levels[actual]
    }

    pub fn flip(&mut self, pos:u32, offset:u32) -> u32 {
        let ref mut levels = self.levels;
        let actual = (pos + offset) as usize;
        match levels[actual] {
            0 => { levels[actual] = 1; 0 }
            _ => 1
        }
    }

    pub fn add(&mut self, at:usize) -> u32 {
        let ix = self.ix;
        for _ in 0..64 {
            self.levels.push(0);
        }
        // println!("      adding {:?} {:?}", ix, at);
        self.levels[at as usize] = ix;
        self.ix = ix + 64;
        ix
    }
}

//-------------------------------------------------------------------------
// BitMatrix
//-------------------------------------------------------------------------

#[derive(Debug)]
pub struct BitMatrix {
    level_store: LevelStore,
    cardinality: u32,
    height: u32,
    bins: u32,
}

impl BitMatrix {
    pub fn new() -> BitMatrix {
        BitMatrix{ level_store: LevelStore::new(), cardinality: 0, height: 5, bins: 8 }
    }

    pub fn size(&self) -> u32 {
        self.bins.pow(self.height)
    }

    pub fn insert(&mut self, e:u32, v:u32) -> bool {
        let bins = self.bins;
        let mut size = self.size();
        let mut e_start = 0;
        let mut v_start = 0;
        let mut current = 0;
        // println!("Starting insert for: {:?} {:?}", e, v);
        for level in 0..self.height - 1 {
            let e_edge = size/bins;
            let v_edge = size/bins;
            let e_ix = (e - e_start) / e_edge;
            let v_ix = (v - v_start) / v_edge;
            let pos = e_ix * bins + v_ix;
            // println!("Inserting at level {:?}: {:?} {:?} | {:?} {:?} | size {:?} pos {:?} {:?}", level, e_ix, v_ix, e_start, v_start, size, pos, current);
            current = self.level_store.fetch(current, pos);
            size = size / bins;
            if e_ix > 0 { e_start = e_start + e_edge * e_ix }
            if v_ix > 0 { v_start = v_start + v_edge * v_ix }
        }
        let e_ix = e - e_start;
        let v_ix = (v - v_start) % bins;
        let pos = (e_ix * bins) + v_ix;
        // println!("Inserting at level {:?}: {:?} {:?} | {:?} {:?} | size {:?}", 4, e_ix, v_ix, e_start, v_start, size);
        // println!("POS {:?} {:?} | {:?} {:?}", current, pos, e_ix, v_ix);
        match self.level_store.flip(current, pos) {
            0 => {
                self.cardinality += 1;
                true
            },
            _ => false,
        }
    }

    pub fn check(&self, e:u32, v:u32) -> bool {
        if e > 0 && v > 0 {
            self.check_both(e, v)
        } else if e > 0 {
            self.check_value(e)
        } else {
            self.check_entity(v)
        }
    }

    pub fn check_both(&self, e:u32, v:u32) -> bool {
        let bins = self.bins;
        let mut size = self.size();
        let mut e_start = 0;
        let mut v_start = 0;
        let mut current = 0;
        for _ in 0..self.height - 1 {
            let e_edge = size/bins;
            let v_edge = size/bins;
            let e_ix = (e - e_start) / e_edge;
            let v_ix = (v - v_start) / v_edge;
            let pos = e_ix * bins + v_ix;
            current = match self.level_store.get(current, pos) {
                0 => return false,
                any => any,
            };
            size = size / bins;
            if e_ix > 0 { e_start = e_start + e_edge * e_ix }
            if v_ix > 0 { v_start = v_start + v_edge * v_ix }
            // println!("Checking at level {:?}: {:?} {:?} | {:?} {:?}", level, e_ix, v_ix, e_start, v_start)
        }
        let e_ix = e - e_start;
        let v_ix = (v - v_start) % bins;
        let pos = (e_ix * bins) + v_ix;
        // println!("CHECKING POS {:?} {:?} {:?} {:?}", current, pos, e_ix, v_ix);
        match self.level_store.get(current, pos) {
            0 => false,
            _ => true,
        }
    }

    pub fn check_entity(&self, v: u32) -> bool {
        let bins = self.bins;
        // Each frame on the stack is encoded as:
        //    level, matrix, e_start, v_start
        let mut queue = vec![0, 0, 0, 0];
        let mut queue_pos = 0;
        let mut queue_length = 1;
        let max_level = self.height - 1;
        let full_size = self.size();
        while queue_pos < queue_length {
            let curPos = queue_pos * 4;
            let level = queue[curPos];
            let matrix = queue[curPos + 1];
            let e_start = queue[curPos + 2];
            let v_start = queue[curPos + 3];
            let size = full_size / bins.pow(level);
            // since only the column is fixed, we need to look at all the rows.
            for e_ix in 0..bins {
                // find the subarray that contain that column and the current row
                let v_edge = v_start + size/bins;
                let v_ix = v / v_edge;
                let e_value = e_start + e_ix * size / bins;
                let pos = e_ix * bins + v_ix;
                let next = self.level_store.get(matrix, pos);
                match next {
                    0 => {},
                    any => {
                        // if we are at the leaves, add this to the fill
                        if level == max_level {
                            return true
                        } else {
                            // if we're not at the leaves, push them onto the stack
                            queue.push(level + 1);
                            queue.push(next);
                            queue.push(e_value);
                            queue.push(v_start + v_ix * (size / bins));
                            queue_length = queue_length + 1;
                        }
                    }
                }
            }

            // now that we've looked at all the rows, we move the queue forward
            queue_pos = queue_pos + 1;
        }
        false
    }

    pub fn check_value(&self, e: u32) -> bool {
        let bins = self.bins;
        // Each frame on the stack is encoded as:
        //    level, matrix, e_start, v_start
        let mut queue = vec![0, 0, 0, 0];
        let mut queue_pos = 0;
        let mut queue_length = 1;
        let max_level = self.height - 1;
        let full_size = self.size();
        while queue_pos < queue_length {
            let curPos = queue_pos * 4;
            let level = queue[curPos];
            let matrix = queue[curPos + 1];
            let e_start = queue[curPos + 2];
            let v_start = queue[curPos + 3];
            let size = full_size / bins.pow(level);
            // since only the column is fixed, we need to look at all the rows.
            for v_ix in 0..bins {

                // find the subarray that contain that row and the current column
                let e_edge = e_start + size/bins;
                let e_ix = e / e_edge;
                let v_value = v_start + v_ix * size / bins;
                let pos = e_ix * bins + v_ix;
                let next = self.level_store.get(matrix, pos);
                // println!("Checking {:?} {:?} {:?} | e_ix {:?} v_ix {:?} | size {:?}", matrix, pos, next, e_ix, v_ix, size);
                match next {
                    0 => {},
                    any => {
                        // if we are at the leaves, add this to the fill
                        if level == max_level {
                            return true;
                        } else {
                            // if we're not at the leaves, push them onto the stack
                            queue.push(level + 1);
                            queue.push(next);
                            queue.push(e_start + e_ix * (size / bins));
                            queue.push(v_value);
                            queue_length = queue_length + 1;
                        }
                    }
                }
            }

            // now that we've looked at all the rows, we move the queue forward
            queue_pos = queue_pos + 1;
        }
        false
    }

    pub fn find_entities(&self, v: u32, fill:&mut Vec<u32>) {
        let bins = self.bins;
        // Each frame on the stack is encoded as:
        //    level, matrix, e_start, v_start
        let mut queue = vec![0, 0, 0, 0];
        let mut queue_pos = 0;
        let mut queue_length = 1;
        let max_level = self.height - 1;
        let full_size = self.size();
        while queue_pos < queue_length {
            let curPos = queue_pos * 4;
            let level = queue[curPos];
            let matrix = queue[curPos + 1];
            let e_start = queue[curPos + 2];
            let v_start = queue[curPos + 3];
            let size = full_size / bins.pow(level);
            // since only the column is fixed, we need to look at all the rows.
            for e_ix in 0..bins {
                // find the subarray that contain that column and the current row
                let v_edge = v_start + size/bins;
                let v_ix = v / v_edge;
                let e_value = e_start + e_ix * size / bins;
                let pos = e_ix * bins + v_ix;
                let next = self.level_store.get(matrix, pos);
                match next {
                    0 => {},
                    any => {
                        // if we are at the leaves, add this to the fill
                        if level == max_level {
                            fill.push(e_value);
                        } else {
                            // if we're not at the leaves, push them onto the stack
                            queue.push(level + 1);
                            queue.push(next);
                            queue.push(e_value);
                            queue.push(v_start + v_ix * (size / bins));
                            queue_length = queue_length + 1;
                        }
                    }
                }
            }

            // now that we've looked at all the rows, we move the queue forward
            queue_pos = queue_pos + 1;
        }
    }

    pub fn find_values(&self, e: u32, fill:&mut Vec<u32>) {
        let bins = self.bins;
        // Each frame on the stack is encoded as:
        //    level, matrix, e_start, v_start
        let mut queue = vec![0, 0, 0, 0];
        let mut queue_pos = 0;
        let mut queue_length = 1;
        let max_level = self.height - 1;
        let full_size = self.size();
        while queue_pos < queue_length {
            let curPos = queue_pos * 4;
            let level = queue[curPos];
            let matrix = queue[curPos + 1];
            let e_start = queue[curPos + 2];
            let v_start = queue[curPos + 3];
            let size = full_size / bins.pow(level);
            // since only the column is fixed, we need to look at all the rows.
            for v_ix in 0..bins {

                // find the subarray that contain that row and the current column
                let e_edge = e_start + size/bins;
                let e_ix = e / e_edge;
                let v_value = v_start + v_ix * size / bins;
                let pos = e_ix * bins + v_ix;
                let next = self.level_store.get(matrix, pos);
                // println!("Checking {:?} {:?} {:?} | e_ix {:?} v_ix {:?} | size {:?}", matrix, pos, next, e_ix, v_ix, size);
                match next {
                    0 => {},
                    any => {
                        // if we are at the leaves, add this to the fill
                        if level == max_level {
                            fill.push(v_value);
                        } else {
                            // if we're not at the leaves, push them onto the stack
                            queue.push(level + 1);
                            queue.push(next);
                            queue.push(e_start + e_ix * (size / bins));
                            queue.push(v_value);
                            queue_length = queue_length + 1;
                        }
                    }
                }
            }

            // now that we've looked at all the rows, we move the queue forward
            queue_pos = queue_pos + 1;
        }
    }

    pub fn all_values(&self, fill:&mut Vec<u32>) {
        let bins = self.bins;
        // Each frame on the stack is encoded as:
        //    level, matrix, e_start, v_start
        let mut queue = vec![0, 0, 0, 0];
        let mut queue_pos = 0;
        let mut queue_length = 1;
        let level_size = bins * bins;
        let max_level = self.height - 1;
        let full_size = self.size();
        while queue_pos < queue_length {
            let curPos = queue_pos * 4;
            let level = queue[curPos];
            let matrix = queue[curPos + 1];
            let e_start = queue[curPos + 2];
            let v_start = queue[curPos + 3];
            let size = full_size / bins.pow(level);
            for pos in 0..level_size {
                let next = self.level_store.get(matrix, pos);
                if next > 0 {
                    let e_ix = pos / bins;
                    let e_value = e_start + e_ix * size / bins;
                    let v_ix = pos % bins;
                    let v_value = v_start + v_ix * size / bins;
                    // if we are at the leaves, add this to the fill
                    if level == max_level {
                        fill.push(e_value);
                    } else {
                        // if we're not at the leaves, push them onto the stack
                        queue.push(level + 1);
                        queue.push(next);
                        queue.push(e_value);
                        queue.push(v_value);
                        queue_length = queue_length + 1;
                    }
                }
            }

            // now that we've looked at all the rows, we move the queue forward
            queue_pos = queue_pos + 1;
        }
    }

    pub fn propose(&self, iter:&mut EstimateIter, e:u32, v:u32) {
        match *iter {
            EstimateIter::Scan { ref mut estimate, ref mut output, ref mut values, pos } => {
                if e > 0 {
                    // println!("here looking for v {:?}", e);
                    self.find_values(e, values);
                    *estimate = values.len() as u32;
                    *output = 2;
                } else if v > 0 {
                    self.find_entities(v, values);
                    *estimate = values.len() as u32;
                    *output = 0;
                } else {
                    self.all_values(values);
                    *estimate = self.cardinality;
                    *output = 0;
                }
            }
        }
    }
}

//-------------------------------------------------------------------------
// BitIndex
//-------------------------------------------------------------------------

// #[derive(Debug)]
pub struct BitIndex {
    matrices: HashMap<u32, BitMatrix>,
    cardinality: u32
}

impl BitIndex {
    pub fn new() -> BitIndex {
        BitIndex { matrices: HashMap::new(), cardinality:0 }
    }
    pub fn insert(&mut self, e:u32, a:u32, v:u32, n:u32, round:u32, transaction:u32, count:u32) -> bool {
        let matrix = self.matrices.entry(a).or_insert_with(|| { BitMatrix::new() });
        matrix.insert(e,v)
    }
    pub fn propose(&self, e:u32, a:u32, v:u32) -> EstimateIter {
        if a == 0 {
            // @FIXME: this isn't always safe. In the case where we have an arbitrary lookup, if we
            // then propose, we might propose values that we then never actually check are correct.
            let mut values = vec![];
            for key in self.matrices.keys() {
                values.push(*key);
            }
            EstimateIter::Scan { estimate: values.len() as u32, pos: 0, values, output: 1 }
        } else {
            let mut iter = EstimateIter::Scan { estimate:0, pos: 0, values: vec![], output: 0 };
            let matrix = match self.matrices.get(&a) {
                None => return iter,
                Some(matrix) => matrix,
            };
            matrix.propose(&mut iter, e, v);
            iter
        }
    }
    pub fn resolve_proposal() {

    }
    pub fn check(&self, e:u32, a:u32, v:u32) -> bool {
        // @FIXME: this isn't always safe. In the case where we have an arbitrary lookup, if we
        // then propose, we might propose values that we then never actually check are correct.
        if a == 0 {
            return true;
        }
        let matrix = match self.matrices.get(&a) {
            None => return false,
            Some(matrix) => matrix,
        };
        matrix.check(e,v)
    }
    pub fn get_diffs() {

    }
    pub fn get() {

    }
}

//-------------------------------------------------------------------------
// Distinct Index
//-------------------------------------------------------------------------


pub struct DistinctIndex {
    index: HashMap<(u32, u32, u32), Vec<i32>>,
}

impl DistinctIndex {
    pub fn new() -> DistinctIndex {
        DistinctIndex { index: HashMap::new() }
    }

    pub fn distinct(&mut self, input:&Change, results:&mut Vec<Change>) {
        let key = (input.e, input.a, input.v);
        let input_count = input.count;
        let mut counts = self.index.entry(key).or_insert_with(|| vec![]);
        println!("Pre counts {:?}", counts);
        ensure_len(counts, (input.round + 1) as usize);
        let counts_len = counts.len() as u32;
        let min = cmp::min(input.round + 1, counts_len);
        let mut cur_count = 0;
        for ix in 0..min {
           cur_count += counts[ix as usize];
        };

        // @TODO: handle Infinity/-Infinity for commits at round 0

        let next_count = cur_count + input_count;
        let delta = get_delta(cur_count, next_count);
        if delta != 0 {
            results.push(input.with_round_count(input.round, delta));
        }

        cur_count = next_count;
        counts[input.round as usize] += input.count;

        for round_ix in (input.round + 1)..counts_len {
            let round_count = counts[round_ix as usize];
            if round_count == 0 { continue; }

            let last_count = cur_count - input_count;
            let next_count = last_count + round_count;
            let delta = get_delta(last_count, next_count);

            let last_count_changed = cur_count;
            let next_count_changed = cur_count + round_count;
            let delta_changed = get_delta(last_count_changed, next_count_changed);

            let mut final_delta = 0;
            if delta != 0 && delta != delta_changed {
                //undo the delta
                final_delta = -delta;
            } else if delta != delta_changed {
                final_delta = delta_changed;
            }

            if final_delta != 0 {
                println!("HERE {:?} {:?} | {:?} {:?}", round_ix, final_delta, delta, delta_changed);
                results.push(input.with_round_count(round_ix, final_delta));
            }

            cur_count = next_count_changed;
        }
        println!("Post counts {:?}", counts);
    }
}

//-------------------------------------------------------------------------
// Distinct tests
//-------------------------------------------------------------------------

#[cfg(test)]
mod DistinctTests {
    extern crate test;

    use super::*;
    use self::test::Bencher;

    fn round_counts_to_changes(counts: Vec<(u32, i32)>) -> Vec<Change> {
        let mut changes = vec![];
        let cur = Change { e: 1, a: 2, v: 3, n: 4, transaction: 1, round: 0, count: 0 };
        for &(round, count) in counts.iter() {
            changes.push(cur.with_round_count(round, count));
        }
        changes
    }

    fn test_distinct(counts: Vec<(u32, i32)>, expected: Vec<(u32, i32)>) {
        let mut index = DistinctIndex::new();
        let changes = round_counts_to_changes(counts);

        let mut final_results: HashMap<u32, i32> = HashMap::new();
        let mut distinct_changes = vec![];
        for change in changes.iter() {
            distinct_changes.clear();
            index.distinct(change, &mut distinct_changes);
            for distinct in distinct_changes.iter() {
                println!("distinct: {:?}", distinct);
                let cur = if final_results.contains_key(&distinct.round) { final_results[&distinct.round] } else { 0 };
                final_results.insert(distinct.round, cur + distinct.count);
            }
        }

        println!("final {:?}", final_results);

        let mut expected_map = HashMap::new();
        for &(round, count) in expected.iter() {
            expected_map.insert(round, count);
            let valid = match final_results.get(&round) {
                Some(&actual) => actual == count,
                None => count == 0,
            };
            assert!(valid, "round {:?} :: expected {:?}, actual {:?}", round, count, final_results.get(&round));
        }

        for (round, count) in final_results.iter() {
            let valid = match expected_map.get(&round) {
                Some(&actual) => actual == *count,
                None => *count == 0,
            };
            assert!(valid, "round {:?} :: expected {:?}, actual {:?}", round, expected_map.get(&round), count);
        }

    }

    #[test]
    fn basic() {
        test_distinct(vec![
            (1,1),
            (2,-1),

            (1, 1),
            (3, -1),
        ], vec![
            (1, 1),
            (3, -1)
        ])
    }

    #[test]
    fn basic_2() {
        test_distinct(vec![
            (1,1),
            (2,-1),

            (3, 1),
            (4, -1),
        ], vec![
            (1, 1),
            (2, -1),
            (3, 1),
            (4, -1),
        ])
    }

    #[test]
    fn basic_2_reverse_order() {
        test_distinct(vec![
            (3,1),
            (4,-1),

            (1, 1),
            (2, -1),
        ], vec![
            (1, 1),
            (2, -1),
            (3, 1),
            (4, -1),
        ])
    }

    #[test]
    fn basic_2_undone() {
        test_distinct(vec![
            (1,1),
            (2,-1),

            (3, 1),
            (4, -1),

            (1,-1),
            (2,1),
        ], vec![
            (3, 1),
            (4, -1),
        ])
    }

    #[test]
    fn basic_multiple() {
        test_distinct(vec![
            (1,1),
            (1,1),
            (1,1),
            (1,1),
            (2,-1),
            (2,-1),
            (2,-1),
            (2,-1),

            (3, 1),
            (3, 1),
            (3, 1),
            (4, -1),
            (4, -1),
            (4, -1),
        ], vec![
            (1, 1),
            (2, -1),
            (3, 1),
            (4, -1),
        ])
    }

    #[test]
    fn basic_multiple_reversed() {
        test_distinct(vec![
            (3, 1),
            (3, 1),
            (3, 1),
            (4, -1),
            (4, -1),
            (4, -1),

            (1,1),
            (1,1),
            (1,1),
            (1,1),
            (2,-1),
            (2,-1),
            (2,-1),
            (2,-1),
        ], vec![
            (1, 1),
            (2, -1),
            (3, 1),
            (4, -1),
        ])
    }

    #[test]
    fn basic_interleaved() {
        test_distinct(vec![
            (3, 1),
            (4, -1),
            (3, 1),
            (4, -1),
            (3, 1),
            (4, -1),

            (1,1),
            (2,-1),
            (1,1),
            (2,-1),
            (1,1),
            (2,-1),
            (1,1),
            (2,-1),
        ], vec![
            (1, 1),
            (2, -1),
            (3, 1),
            (4, -1),
        ])
    }

    #[test]
    fn basic_multiple_negative_first() {
        test_distinct(vec![
            (2,-1),
            (2,-1),
            (2,-1),
            (1,1),
            (1,1),
            (1,1),

            (4, -1),
            (4, -1),
            (4, -1),
            (3, 1),
            (3, 1),
            (3, 1),
        ], vec![
            (1, 1),
            (2, -1),
            (3, 1),
            (4, -1),
        ])
    }

    #[test]
    fn basic_multiple_undone() {
        test_distinct(vec![
            (1,1),
            (1,1),
            (1,1),
            (1,1),
            (2,-1),
            (2,-1),
            (2,-1),
            (2,-1),

            (3, 1),
            (3, 1),
            (3, 1),
            (4, -1),
            (4, -1),
            (4, -1),

            (1,-1),
            (1,-1),
            (1,-1),
            (1,-1),
            (2,1),
            (2,1),
            (2,1),
            (2,1),
        ], vec![
            (3, 1),
            (4, -1),
        ])
    }

    #[test]
    fn basic_multiple_undone_interleaved() {
        test_distinct(vec![
            (1,1),
            (1,1),
            (1,1),
            (1,1),
            (2,-1),
            (2,-1),
            (2,-1),
            (2,-1),

            (1,-1),
            (1,-1),
            (1,-1),
            (1,-1),

            (3, 1),
            (3, 1),
            (3, 1),
            (4, -1),
            (4, -1),
            (4, -1),

            (2,1),
            (2,1),
            (2,1),
            (2,1),
        ], vec![
            (3, 1),
            (4, -1),
        ])
    }

    #[test]
    fn basic_multiple_different_counts() {
        test_distinct(vec![
            (1,1),
            (1,1),
            (1,1),
            (1,1),
            (2,-1),
            (2,-1),
            (2,-1),
            (2,-1),

            (3, 1),
            (4, -1),
        ], vec![
            (1, 1),
            (2, -1),
            (3, 1),
            (4, -1),
        ])
    }

    #[test]
    fn basic_multiple_different_counts_extra_removes() {
        test_distinct(vec![
            (1,1),
            (1,1),
            (1,1),
            (1,1),
            (2,-1),
            (2,-1),
            (2,-1),
            (2,-1),

            (1,-1),
            (1,-1),
            (1,-1),
            (1,-1),
            (2,1),
            (2,1),
            (2,1),
            (2,1),

            (3, 1),
            (4, -1),
        ], vec![
            (3, 1),
            (4, -1),
        ])
    }

    #[test]
    fn simple_round_promotion() {
        test_distinct(vec![
            (8,1),
            (9,-1),

            (5,1),
            (6,-1),
            (8,-1),
            (9,1),
        ], vec![
            (5, 1),
            (6, -1)
        ])
    }

    #[test]
    fn full_promotion() {
        test_distinct(vec![
            (9,1),
            (9,1),
            (10,-1),
            (10,-1),

            (9,1),
            (9,1),
            (10,-1),
            (10,-1),

            (9,-1),
            (10,1),
            (9,-1),
            (10,1),

            (9,-1),
            (10,1),
            (9,-1),
            (10,1),
        ], vec![
            (9, 0),
            (10, 0)
        ])
    }

    #[test]
    fn positive_full_promotion() {
        test_distinct(vec![
            (7,1),
            (8,-1),
            (8,1),
            (7,1),
            (8,-1),
            (4,1),
            (8, -1),
            (7, 1),
            (8, -1),
            (8, 1),
            (5, -1),
            (7, -3),
            (8, 1),
            (8, 3),
            (5, 1),
            (8, 1),
            (8, -2),
            (8, -1),
        ], vec![
            (4, 1),
        ])
    }
}

//-------------------------------------------------------------------------
// BitIndex Tests
//-------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use self::test::Bencher;
    use std::num::Wrapping;

    #[test]
    fn basic() {
        let mut index = BitIndex::new();
        index.insert(1,1,1,0,0,0,0);
        index.insert(1,2,1,0,0,0,0);
        index.insert(2,3,1,0,0,0,0);
        index.insert(1,3,100,0,0,0,0);
        assert!(index.check(1,1,1));
        assert!(index.check(1,2,1));
        assert!(index.check(2,3,1));
        assert!(index.check(1,3,100));
        assert!(!index.check(100,300,100));
    }

    #[test]
    fn basic2() {
        let mut index = BitIndex::new();
        index.insert(5,3,8,0,0,0,0);
        index.insert(9,3,8,0,0,0,0);
        assert!(index.check(5,3,8));
        assert!(index.check(9,3,8));
        assert!(!index.check(100,300,100));
    }

    #[test]
    fn find_entities() {
        let mut index = BitMatrix::new();
        index.insert(1,1);
        index.insert(2,1);
        index.insert(300,1);
        let mut entities = vec![];
        index.find_entities(1, &mut entities);
        assert!(entities.contains(&1));
        assert!(entities.contains(&2));
        assert!(entities.contains(&300));
        assert!(!entities.contains(&3));
    }

    #[test]
    fn find_values() {
        let mut index = BitMatrix::new();
        // index.insert(1,1);
        // index.insert(1,2);
        // index.insert(1,300);
        // let mut values = vec![];
        // index.find_values(1, &mut values);
        // assert!(values.contains(&1));
        // assert!(values.contains(&2));
        // assert!(values.contains(&300));
        // assert!(!values.contains(&3));

        index.insert(5,8);
        index.insert(9,8);
        let mut values2 = vec![];
        index.find_values(9, &mut values2);
        assert!(values2.contains(&8));
    }

     #[test]
    fn basic_propose() {
        let mut index = BitIndex::new();
        index.insert(1,1,1,0,0,0,0);
        index.insert(2,1,1,0,0,0,0);
        index.insert(2,1,7,0,0,0,0);
        index.insert(3,1,1,0,0,0,0);
        index.insert(2,3,1,0,0,0,0);
        index.insert(1,3,100,0,0,0,0);
        let proposal1 = index.propose(0,1,1);
        assert_eq!(proposal1.estimate(), 3);
        let proposal2 = index.propose(2,1,0);
        assert_eq!(proposal2.estimate(), 2);
    }


    fn rand(rseed:u32) -> u32 {
        return ((Wrapping(rseed) * Wrapping(1103515245) + Wrapping(12345)) & Wrapping(0x7fffffff)).0;
    }

    #[bench]
    fn benchmark(b:&mut Bencher) {
        let mut total = 0;
        let mut times = 0;
        let mut levels = 0;
        b.iter(|| {
            times += 1;
            let mut index = BitMatrix::new();
            let mut seed = 0;
            for ix in 0..100000 {
                let e = rand(seed + ix);
                seed = e;
                let val = rand(seed + ix);
                seed = val;
                index.insert(e % 10000, val % 10000);
            }
            total += index.cardinality;
            levels = index.level_store.levels.len() / 64;
        });
        println!("{:?} : {:?}", times, total);
        println!("levels: {:?}", levels);
    }

    #[bench]
    fn benchmark_read(b:&mut Bencher) {
        let mut total = 0;
        let mut times = 0;
        let mut levels = 0;
        let mut index = BitMatrix::new();
        let mut seed = 0;
        for ix in 0..100000 {
            let e = rand(seed);
            seed = e;
            let val = rand(seed);
            seed = val;
            index.insert(e % 10000, val % 10000);
        }
        seed = 0;
        let mut v = vec![];
        b.iter(|| {
            let e = rand(seed);
            seed = e;
            let val = rand(seed);
            seed = val;
            index.find_values(e % 10000, &mut v);
        });
        println!("{:?} : {:?}", times, total);
        println!("found: {:?}", v.len());
    }

    struct HashIndex {
        full: HashMap<(u32, u32), bool>,
        e: HashMap<u32, Vec<u32>>,
        v: HashMap<u32, Vec<u32>>,
        size: u32,
    }

    impl HashIndex {
        pub fn new() -> HashIndex {
            HashIndex { full: HashMap::with_capacity(100000), e: HashMap::new(), v: HashMap::new(), size: 0 }
        }
        // pub fn insert(&mut self, e: u32, v:u32) -> bool {

        //     let key = (e, v);
        //     let existed = self.full.contains_key(&key);
        //     if !existed {
        //         self.size += 1;
        //         self.full.insert(key, true);
        //         let es = self.e.entry(e).or_insert_with(|| vec![]);
        //         es.push(v);
        //         let vs = self.v.entry(v).or_insert_with(|| vec![]);
        //         vs.push(e);
        //     };
        //     existed
        // }

        pub fn insert(&mut self, e: u32, v:u32) -> bool {
            let added = match self.e.entry(e) {
                Entry::Occupied(mut o) => {
                    let mut es = o.get_mut();
                    if es.contains(&v) {
                        false
                    } else {
                        es.push(v);
                        true
                    }
                }
                Entry::Vacant(o) => {
                    o.insert(vec![v]);
                    true
                },
            };
            if added {
                let vs = self.v.entry(v).or_insert_with(|| vec![]);
                vs.push(e);
            }
            added
        }

        pub fn check(&self, e: u32, v:u32) -> bool {
            let key = (e, v);
            self.full.contains_key(&key)
        }

        pub fn find_values(&self, e:u32, vec: &mut Vec<u32>) {
            match self.e.get(&e) {
                Some(es) => vec.extend(es),
                None => {},
            }
        }
    }

    #[bench]
    fn bench_hash_write(b:&mut Bencher) {
        let mut total = 0;
        let mut times = 0;
        b.iter(|| {
            times += 1;
            let mut index = HashIndex::new();
            let mut seed = 0;
            for ix in 0..100000 {
                let e = rand(seed);
                seed = e;
                let val = rand(seed);
                seed = val;
                index.insert(e % 10000, val % 10000);
                // println!("inserting {:?} {:?}", e, val);
            }
            // total += index.size;
        });
        println!("{:?} : {:?}", times, total);
    }

    // #[bench]
    // fn benchmark_hash_read(b:&mut Bencher) {
    //     let mut total = 0;
    //     let mut times = 0;
    //     let mut levels = 0;
    //     let mut index = HashIndex::new();
    //     let mut seed = 0;
    //     for ix in 0..100000 {
    //         let e = rand(seed);
    //         seed = e;
    //         let val = rand(seed);
    //         seed = val;
    //         index.insert(e % 10000, val % 10000);
    //     }
    //     seed = 0;
    //     // let mut v = vec![];
    //     b.iter(|| {
    //         let e = rand(seed);
    //         seed = e;
    //         let val = rand(seed);
    //         seed = val;
    //         index.check(e % 10000, val % 10000);
    //     });
    //     // println!("results: {:?}", v.len());
    // }



    #[bench]
    fn bench_faster_radix_sort_small(b: &mut Bencher) {
        // let data = vec![1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,4];
        // let data = generate_unsorted(20);
        let mut data = vec![];
        let mut seed = 0;
        for _ in 0..1000 {
            let e = rand(seed);
            seed = e;
            // let val = rand(seed);
            // seed = val;
            data.push(e);
        }
        // data.sort();
        b.iter(|| {
            // let mut foo = data.clone();
            // faster_radix_sort(&mut foo);
            for _ in 0..5 {
                let e = rand(seed);
                seed = e;
                data.insert((e % 1000) as usize, 0);
            }
            // data.sort();
        })
    }
}

