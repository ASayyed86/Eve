//-------------------------------------------------------------------------
// Indexes
//-------------------------------------------------------------------------

use std::collections::HashMap;
use ops::{EstimateIter, Change, RoundHolder};
use std::cmp;
use std;
use std::collections::hash_map::Entry;

extern crate fnv;
use indexes::fnv::FnvHasher;
use std::hash::BuildHasherDefault;

type MyHasher = BuildHasherDefault<FnvHasher>;

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
// HashIndexLevel
//-------------------------------------------------------------------------

pub struct HashIndexLevel {
    e: HashMap<u32, Vec<u32>, MyHasher>,
    v: HashMap<u32, Vec<u32>, MyHasher>,
    size: u32,
}

impl HashIndexLevel {
    pub fn new() -> HashIndexLevel {
        HashIndexLevel { e: HashMap::default(), v: HashMap::default(), size: 0 }
    }

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
            self.size += 1;
            let vs = self.v.entry(v).or_insert_with(|| vec![]);
            vs.push(e);
        }
        added
    }

    pub fn check(&self, e: u32, v:u32) -> bool {
        if e > 0 && v > 0 {
            match self.e.get(&e) {
                Some(es) => es.contains(&v),
                None => false,
            }
        } else if e > 0 {
            self.e.contains_key(&e)
        } else if v > 0 {
            self.v.contains_key(&v)
        } else {
            self.size > 0
        }
    }

    pub fn find_values(&self, e:u32, vec: &mut Vec<u32>) {
        match self.e.get(&e) {
            Some(vs) => vec.extend(vs),
            None => {},
        }
    }

    pub fn find_entities(&self, v:u32, vec: &mut Vec<u32>) {
        match self.v.get(&v) {
            Some(es) => vec.extend(es),
            None => {},
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
                    if self.e.len() < self.v.len() {
                        for key in self.e.keys() {
                            values.push(*key);
                        }
                        *output = 0;
                    } else {
                        for key in self.v.keys() {
                            values.push(*key);
                        }
                        *output = 2;
                    }
                    *estimate = values.len() as u32;
                }
            }
        }
    }
}

//-------------------------------------------------------------------------
// HashIndex
//-------------------------------------------------------------------------

pub struct HashIndex {
    a: HashMap<u32, HashIndexLevel, MyHasher>,
    size: u32,
}

impl HashIndex {
    pub fn new() -> HashIndex{
        HashIndex { a: HashMap::default(), size: 0 }
    }

    pub fn insert(&mut self, e: u32, a:u32, v:u32) -> bool {
        let added = match self.a.entry(a) {
            Entry::Occupied(mut o) => {
                let mut level = o.get_mut();
                level.insert(e, v)
            }
            Entry::Vacant(o) => {
                let mut level = HashIndexLevel::new();
                level.insert(e,v);
                o.insert(level);
                true
            },
        };
        if added { self.size += 1; }
        added
    }

    pub fn check(&self, e: u32, a:u32, v:u32) -> bool {
        if a > 0 {
            match self.a.get(&a) {
                Some(level) => level.check(e, v),
                None => false,
            }
        } else {
            panic!("Haven't implemented check for free a")
        }
    }

    pub fn propose(&self, e:u32, a:u32, v:u32) -> EstimateIter {
        if a == 0 {
            // @FIXME: this isn't always safe. In the case where we have an arbitrary lookup, if we
            // then propose, we might propose values that we then never actually check are correct.
            let mut values = vec![];
            for key in self.a.keys() {
                values.push(*key);
            }
            EstimateIter::Scan { estimate: values.len() as u32, pos: 0, values, output: 1 }
        } else {
            let mut iter = EstimateIter::Scan { estimate:0, pos: 0, values: vec![], output: 0 };
            let level = match self.a.get(&a) {
                None => return iter,
                Some(level) => level,
            };
            level.propose(&mut iter, e, v);
            iter
        }
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

    pub fn distinct(&mut self, input:&Change, rounds:&mut RoundHolder) {
        let key = (input.e, input.a, input.v);
        let input_count = input.count;
        let mut counts = self.index.entry(key).or_insert_with(|| vec![]);
        // println!("Pre counts {:?}", counts);
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
            rounds.insert(input.with_round_count(input.round, delta));
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
                // println!("HERE {:?} {:?} | {:?} {:?}", round_ix, final_delta, delta, delta_changed);
                rounds.insert(input.with_round_count(round_ix, final_delta));
            }

            cur_count = next_count_changed;
        }
        // println!("Post counts {:?}", counts);
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
        let mut distinct_changes = RoundHolder::new();
        for change in changes.iter() {
            index.distinct(change, &mut distinct_changes);
        }
        for distinct in distinct_changes.iter() {
            println!("distinct: {:?}", distinct);
            let cur = if final_results.contains_key(&distinct.round) { final_results[&distinct.round] } else { 0 };
            final_results.insert(distinct.round, cur + distinct.count);
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
// HashIndex Tests
//-------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use self::test::Bencher;
    use std::num::Wrapping;

    #[test]
    fn basic() {
        let mut index = HashIndex::new();
        index.insert(1,1,1);
        index.insert(1,2,1);
        index.insert(2,3,1);
        index.insert(1,3,100);
        assert!(index.check(1,1,1));
        assert!(index.check(1,2,1));
        assert!(index.check(2,3,1));
        assert!(index.check(1,3,100));
        assert!(!index.check(100,300,100));
    }

    #[test]
    fn basic2() {
        let mut index = HashIndex::new();
        index.insert(5,3,8);
        index.insert(9,3,8);
        assert!(index.check(5,3,8));
        assert!(index.check(9,3,8));
        assert!(!index.check(100,300,100));
    }

    #[test]
    fn find_entities() {
        let mut index = HashIndexLevel::new();
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
        let mut index = HashIndexLevel::new();
        index.insert(1,1);
        index.insert(1,2);
        index.insert(1,300);
        let mut values = vec![];
        index.find_values(1, &mut values);
        assert!(values.contains(&1));
        assert!(values.contains(&2));
        assert!(values.contains(&300));
        assert!(!values.contains(&3));

        index.insert(5,8);
        index.insert(9,8);
        let mut values2 = vec![];
        index.find_values(9, &mut values2);
        assert!(values2.contains(&8));
    }

     #[test]
    fn basic_propose() {
        let mut index = HashIndex::new();
        index.insert(1,1,1);
        index.insert(2,1,1);
        index.insert(2,1,7);
        index.insert(3,1,1);
        index.insert(2,3,1);
        index.insert(1,3,100);
        let proposal1 = index.propose(0,1,1);
        assert_eq!(proposal1.estimate(), 3);
        let proposal2 = index.propose(2,1,0);
        assert_eq!(proposal2.estimate(), 2);
    }


    fn rand(rseed:u32) -> u32 {
        return ((Wrapping(rseed) * Wrapping(1103515245) + Wrapping(12345)) & Wrapping(0x7fffffff)).0;
    }


    #[bench]
    fn bench_hash_write(b:&mut Bencher) {
        let mut total = 0;
        let mut times = 0;
        let mut index = HashIndex::new();
        let mut seed = 0;
        // for ix in 0..10_000_000 {
        //     let e = rand(seed);
        //     seed = e;
        //     let a = rand(seed);
        //     seed = a;
        //     let val = rand(seed);
        //     seed = val;
        //     index.insert(e % 10000, (a % 50) + 1, val % 10000);
        // }
        seed = 0;
        b.iter(|| {
            times += 1;
            let e = rand(seed);
            seed = e;
            let a = rand(seed);
            seed = a;
            let val = rand(seed);
            seed = val;
            index.insert(e % 100000, (a % 50) + 1, val % 100000);
            // if(index.size > 100000) {
            //     index = HashIndex3::new();
            // }
            // total += index.size;
        });
        println!("{:?} : {:?}", times, index.size);
    }

    #[bench]
    fn bench_hash_read(b:&mut Bencher) {
        let mut total = 0;
        let mut times = 0;
        let mut levels = 0;
        let mut index = HashIndex::new();
        let mut seed = 0;
        for ix in 0..100_000 {
            let e = rand(seed);
            seed = e;
            let a = rand(seed);
            seed = a;
            let val = rand(seed);
            seed = val;
            index.insert(e % 100000, (a % 50) + 1, val % 100000);
        }
        seed = 0;
        // let mut v = vec![];
        b.iter(|| {
            let e = rand(seed);
            seed = e;
            let a = rand(seed);
            seed = a;
            let val = rand(seed);
            seed = val;
            total += seed;
            index.check(e % 100000, (a % 50) + 1, val % 100000);
        });
        println!("results: {:?}", total);
    }



}

