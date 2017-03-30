use std::collections::HashMap;

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
        let next_ix = ix + 64;
        for _ in 0..64 {
            self.levels.push(0);
        }
        self.levels[at as usize] = ix;
        self.ix = next_ix;
        ix
    }
}

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
        self.cardinality = self.cardinality + 1;
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
            current = self.level_store.fetch(current, pos);
            size = size / bins;
            if e_ix > 0 { e_start = e_start + e_edge * e_ix }
            if v_ix > 0 { v_start = v_start + v_edge * v_ix }
            // println!("Inserting at level {:?}: {:?} {:?} | {:?} {:?}", level, e_ix, v_ix, e_start, v_start)
        }
        let e_ix = e - e_start;
        let v_ix = (v - v_start) % bins;
        let pos = (e_ix * bins) + v_ix;
        // println!("POS {:?} {:?} {:?} {:?}", current, pos, e_ix, v_ix);
        match self.level_store.flip(current, pos) {
            0 => true,
            _ => false,
        }
    }

    pub fn check(&self, e:u32, v:u32) -> bool {
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
}

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
    pub fn propose() {

    }
    pub fn resolve_proposal() {

    }
    pub fn check(&mut self, e:u32, a:u32, v:u32, n:u32, round:u32, transaction:u32, count:u32) -> bool {
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
        assert!(index.check(1,1,1,0,0,0,0));
        assert!(index.check(1,2,1,0,0,0,0));
        assert!(index.check(2,3,1,0,0,0,0));
        assert!(index.check(1,3,100,0,0,0,0));
        assert!(!index.check(100,300,100,0,0,0,0));
    }

    fn rand(rseed:u32) -> u32 {
        return ((Wrapping(rseed) * Wrapping(1103515245) + Wrapping(12345)) & Wrapping(0x7fffffff)).0;
    }

    #[bench]
    fn benchmark(b:&mut Bencher) {
        b.iter(|| {
            let mut index = BitMatrix::new();
            let mut seed = 0;
            for _ in 0..100000 {
                let e = rand(seed) % 10000;
                seed = e;
                let val = rand(seed) % 10000;
                seed = val;
                index.insert(e, val);
            }

        })
    }
}

