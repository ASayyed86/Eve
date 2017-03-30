
#![feature(test)]
#![feature(link_args)]

// #[link_args = "-s EXPORTED_FUNCTIONS=['_coolrand','_makeIter','_next']"]
extern {}

use std::collections::HashMap;
use std::cmp::Ordering;
use std::time::Instant;
use std::num::Wrapping;

extern crate test;
use test::Bencher;

extern crate rand;
use rand::Rng;

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
        let nextIx = ix + 64;
        for pos in 0..64 {
            self.levels.push(0);
        }
        self.levels[at as usize] = ix;
        self.ix = nextIx;
        ix
    }
}

#[derive(Debug)]
struct BitMatrix {
    levelStore: LevelStore,
    cardinality: u32,
    height: u32,
    bins: u32,
}

impl BitMatrix {
    fn new() -> BitMatrix {
        BitMatrix{ levelStore: LevelStore::new(), cardinality: 0, height: 5, bins: 8 }
    }

    pub fn size(&self) -> u32 {
        self.bins.pow(self.height)
    }

    pub fn insert(&mut self, e:u32, v:u32) -> bool {
        self.cardinality = self.cardinality + 1;
        let bins = self.bins;
        let mut size = self.size();
        let mut eStart = 0;
        let mut vStart = 0;
        let mut current = 0;
        for level in 0..self.height - 1 {
            let eEdge = size/bins;
            let vEdge = size/bins;
            let eIx = (e - eStart) / eEdge;
            let vIx = (v - vStart) / vEdge;
            let pos = eIx * bins + vIx;
            current = self.levelStore.fetch(current, pos);
            size = size / bins;
            if eIx > 0 { eStart = eStart + eEdge * eIx }
            if vIx > 0 { vStart = vStart + vEdge * vIx }
            // println!("Inserting at level {:?}: {:?} {:?} | {:?} {:?}", level, eIx, vIx, eStart, vStart)
        }
        let eIx = e - eStart;
        let vIx = (v - vStart) % bins;
        let pos = (eIx * bins) + vIx;
        // println!("POS {:?} {:?} {:?} {:?}", current, pos, eIx, vIx);
        match self.levelStore.flip(current, pos) {
            0 => true,
            any => false,
        }
    }

    pub fn check(&self, e:u32, v:u32) -> bool {
        let bins = self.bins;
        let mut size = self.size();
        let mut eStart = 0;
        let mut vStart = 0;
        let mut current = 0;
        for level in 0..self.height - 1 {
            let eEdge = size/bins;
            let vEdge = size/bins;
            let eIx = (e - eStart) / eEdge;
            let vIx = (v - vStart) / vEdge;
            let pos = eIx * bins + vIx;
            current = match self.levelStore.get(current, pos) {
                0 => return false,
                any => any,
            };
            size = size / bins;
            if eIx > 0 { eStart = eStart + eEdge * eIx }
            if vIx > 0 { vStart = vStart + vEdge * vIx }
            // println!("Checking at level {:?}: {:?} {:?} | {:?} {:?}", level, eIx, vIx, eStart, vStart)
        }
        let eIx = e - eStart;
        let vIx = (v - vStart) % bins;
        let pos = (eIx * bins) + vIx;
        // println!("CHECKING POS {:?} {:?} {:?} {:?}", current, pos, eIx, vIx);
        match self.levelStore.get(current, pos) {
            0 => false,
            any => true,
        }
    }
}

// #[derive(Debug)]
struct BitIndex {
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
    pub fn resolveProposal() {

    }
    pub fn check(&mut self, e:u32, a:u32, v:u32, n:u32, round:u32, transaction:u32, count:u32) -> bool {
        let matrix = match self.matrices.get(&a) {
            None => return false,
            Some(matrix) => matrix,
        };
        matrix.check(e,v)
    }
    pub fn getDiffs() {

    }
    pub fn get() {

    }
}

fn rand(rseed:u32) -> u32 {
	return ((Wrapping(rseed) * Wrapping(1103515245) + Wrapping(12345)) & Wrapping(0x7fffffff)).0;
}

#[no_mangle]
pub extern fn coolrand(rseed:u32) -> u32 {
	return (rseed * 1103515245 + 12345) & 0x7fffffff;
}

pub struct Iter {
    ix: u32,
}

#[no_mangle]
pub extern fn makeIter() -> Box<Iter> {
    return Box::new(Iter {ix: 0})
}

#[no_mangle]
pub extern fn next(iter:&mut Iter) -> u32 {
    iter.ix = iter.ix + 1;
    iter.ix
}

fn main() {
    let mut index = BitIndex::new();
    index.insert(1,1,1,0,0,0,0);
    index.insert(1,2,1,0,0,0,0);
    index.insert(2,3,1,0,0,0,0);
    index.insert(1,3,100,0,0,0,0);
    println!("{:?}", index.check(1,1,1,0,0,0,0));
    println!("{:?}", index.check(1,2,1,0,0,0,0));
    println!("{:?}", index.check(2,3,1,0,0,0,0));
    println!("{:?}", index.check(1,3,100,0,0,0,0));
    println!("{:?}", index.check(1,9,100,0,0,0,0));

    let start = Instant::now();
    for foo in 0..10 {
        let mut index = BitMatrix::new();
        let mut seed = 0;
        for ix in 0..100000 {
            let e = rand(seed) % 10000;
            seed = e;
            let val = rand(seed) % 10000;
            seed = val;
            index.insert(e, val);
        }

    }
    println!("TOOK {:?}", start.elapsed());
}

#[bench]
fn benchmark(b:&mut Bencher) {
    let mut rng = rand::thread_rng();
    b.iter(|| {
        let mut index = BitMatrix::new();
        let mut seed = 0;
        for ix in 0..100000 {
            let e = rand(seed) % 10000;
            seed = e;
            let val = rand(seed) % 10000;
            seed = val;
            index.insert(e, val);
        }

    })
}
