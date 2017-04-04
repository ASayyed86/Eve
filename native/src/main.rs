#![feature(test)]
#![feature(link_args)]

// #[link_args = "-s EXPORTED_FUNCTIONS=['_coolrand','_makeIter','_next']"]
extern {}
use std::num::Wrapping;
use std::time::Instant;

mod ops;

mod indexes;
use indexes::{BitIndex, BitMatrix};

fn rand(rseed:u32) -> u32 {
	return ((Wrapping(rseed) * Wrapping(1103515245) + Wrapping(12345)) & Wrapping(0x7fffffff)).0;
}

fn main() {
    let mut index = BitIndex::new();
    index.insert(1,1,1,0,0,0,0);
    index.insert(1,2,1,0,0,0,0);
    index.insert(2,3,1,0,0,0,0);
    index.insert(1,3,100,0,0,0,0);
    println!("{:?}", index.check(1,1,1));
    println!("{:?}", index.check(1,2,1));
    println!("{:?}", index.check(2,3,1));
    println!("{:?}", index.check(1,3,100));
    println!("{:?}", index.check(1,9,100));

    let start = Instant::now();
    for _ in 0..10 {
        let mut index = BitMatrix::new();
        let mut seed = 0;
        for _ in 0..100000 {
            let e = rand(seed) % 10000;
            seed = e;
            let val = rand(seed) % 10000;
            seed = val;
            index.insert(e, val);
        }

    }
    println!("TOOK {:?}", start.elapsed());
}
