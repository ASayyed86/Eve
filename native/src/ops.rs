//-------------------------------------------------------------------------
// Ops
//-------------------------------------------------------------------------

// TODO:
//  - hookup distinct
//  - get multiplicities
//  - index insert
//  - functions

use indexes::{HashIndex, DistinctIndex, DistinctIter};
use std::collections::HashMap;
use std::mem::transmute;
use std::time::Instant;
use std::collections::hash_map::Entry;
use std::cmp;

//-------------------------------------------------------------------------
// Change
//-------------------------------------------------------------------------

#[derive(Debug, Copy, Clone)]
pub struct Change {
    pub e: u32,
    pub a: u32,
    pub v: u32,
    pub n: u32,
    pub round: u32,
    pub transaction: u32,
    pub count: i32,
}

impl Change {
    pub fn with_round_count(&self, round: u32, count:i32) -> Change {
        Change {e: self.e, a: self.a, v: self.v, n: self.n, round, transaction: self.transaction, count}
    }
}

//-------------------------------------------------------------------------
// Block
//-------------------------------------------------------------------------

pub struct Block {
    name: String,
    constraints: Vec<Constraint>,
}

//-------------------------------------------------------------------------
// row
//-------------------------------------------------------------------------

#[derive(Debug)]
pub struct Row {
    fields: Vec<u32>,
    count: u32,
    round: u32,
    solved_fields: u64,
    solving_for:u64,
}

impl Row {
    pub fn new(size:usize) -> Row {
        Row { fields: vec![0; size], count: 0, round: 0, solved_fields: 0, solving_for: 0 }
    }

    pub fn set(&mut self, field_index:u32, value:u32) {
        self.fields[field_index as usize] = value;
        self.solving_for = set_bit(0, field_index);
        self.solved_fields = set_bit(self.solved_fields, field_index);
    }

    pub fn clear(&mut self, field_index:u32) {
        self.fields[field_index as usize] = 0;
        self.solving_for = 0;
        self.solved_fields = clear_bit(self.solved_fields, field_index);
    }

    pub fn reset(&mut self, size:u32) {
        self.count = 0;
        self.round = 0;
        self.solved_fields = 0;
        self.solving_for = 0;
        for field_index in 0..size {
            self.fields[field_index as usize] = 0;
        }
    }
}

//-------------------------------------------------------------------------
// Estimate Iter
//-------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum EstimateIter {
    Scan {estimate: u32, pos: u32, values: Vec<u32>, output: u32},
    // Function {estimate: u32, args:Vec<Value>, func: fn(args:Vec<Value>), output: u32},
}

impl EstimateIter {
    pub fn estimate(&self) -> u32 {
        match self {
            &EstimateIter::Scan {estimate, pos, ref values, output} => {
                estimate
            },
        }
    }

    pub fn next(&mut self, row:&mut Row) -> bool {
        match self {
            &mut EstimateIter::Scan {ref estimate, ref mut pos, ref values, ref output} => {
                if *pos >= values.len() as u32 {
                    false
                } else {
                    row.set(*output, values[*pos as usize]);
                    *pos = *pos + 1;
                    true
                }
            },
        }
    }

    pub fn clear(&mut self, row:&mut Row) {
        match self {
            &mut EstimateIter::Scan {ref mut estimate, ref mut pos, ref values, ref output} => {
                row.clear(*output);
            },
        }
    }
}

//-------------------------------------------------------------------------
// Frame
//-------------------------------------------------------------------------

pub struct Frame<'a> {
    input: &'a Change,
    row: Row,
    index: &'a mut HashIndex,
    constraints: Option<&'a Vec<Constraint>>,
    blocks: &'a Vec<Block>,
    iters: Vec<Option<EstimateIter>>,
    distinct: &'a mut DistinctIndex,
    rounds: &'a mut RoundHolder,
}

impl<'a> Frame<'a> {
    pub fn new(index: &'a mut HashIndex, rounds: &'a mut RoundHolder, distinct: &'a mut DistinctIndex, blocks: &'a Vec<Block>, input:&'a Change) -> Frame<'a> {
        Frame {row: Row::new(64), index, rounds, distinct, input, blocks, constraints: None, iters: vec![None; 64]}
    }

    pub fn resolve(&self, field:&Field) -> u32 {
        match field {
            &Field::Register(cur) => self.row.fields[cur],
            &Field::Value(cur) => cur,
        }
    }

    pub fn check_iter(&mut self, iter_ix:u32, iter: EstimateIter) {
        let ix = iter_ix as usize;
        let neue = match self.iters[ix] {
            None => {
                Some(iter)
            },
            Some(ref cur) if cur.estimate() > iter.estimate() => {
                Some(iter)
            },
            _ => None,
        };
        match neue {
            Some(_) => { self.iters[ix] = neue; },
            None => {},
        }
    }
}



//-------------------------------------------------------------------------
// Instruction
//-------------------------------------------------------------------------

pub enum Instruction {
    start_block { block: u32 },
    get_iterator {iterator: u32, bail: i32, constraint: u32},
    iterator_next {iterator: u32, bail: i32},
    accept {bail: i32, constraint:u32},
    get_rounds {bail: i32, constraint: u32},
    output {next: i32, constraint:u32},
}

pub fn start_block(frame: &mut Frame, block:u32) -> i32 {
    // println!("STARTING! {:?}", block);
    frame.constraints = Some(&frame.blocks[block as usize].constraints);
    1
}

pub fn get_iterator(frame: &mut Frame, iter_ix:u32, constraint:u32, bail:i32) -> i32 {
    let cur = match frame.constraints {
        Some(ref constraints) => &constraints[constraint as usize],
        None => return bail,
    };
    match cur {
        &Constraint::Scan {ref e, ref a, ref v, ref register_mask} => {
            // if we have already solved all of this scan's vars, we just move on
            if check_bits(frame.row.solved_fields, *register_mask) {
                return 1;
            }

            let resolved_e = frame.resolve(e);
            let resolved_a = frame.resolve(a);
            let resolved_v = frame.resolve(v);

            // println!("Getting proposal for {:?} {:?} {:?}", resolved_e, resolved_a, resolved_v);
            let mut iter = frame.index.propose(resolved_e, resolved_a, resolved_v);
            match iter {
                EstimateIter::Scan {estimate, pos, ref values, ref mut output} => {
                    *output = match (*output, e, a, v) {
                        (0, &Field::Register(reg), _, _) => reg as u32,
                        (1, _, &Field::Register(reg), _) => reg as u32,
                        (2, _, _, &Field::Register(reg)) => reg as u32,
                        _ => panic!("bad scan output"),
                    };
                }
            }
            frame.check_iter(iter_ix, iter);
            // println!("get scan iterator {:?} : count {:?}", cur, frame.index.propose(resolved_e, resolved_a, resolved_v,0,0,0,0));
        },
        &Constraint::Function {ref op, ref out, ref params} => {
            // println!("get function iterator {:?}", cur);
        },
        _ => {}
    };
    1
}

pub fn iterator_next(frame: &mut Frame, iterator:u32, bail:i32) -> i32 {
    let go = {
        let mut iter = frame.iters[iterator as usize].as_mut();
        // println!("Iter Next: {:?}", iter);
        match iter {
            Some(ref mut cur) => {
                match cur.next(&mut frame.row) {
                    false => {
                        cur.clear(&mut frame.row);
                        bail
                    },
                    true => 1,
                }
            },
            None => bail,
        }
    };
    if go == bail {
        frame.iters[iterator as usize] = None;
    }
    // println!("Row: {:?}", &frame.row.fields[0..3]);
    go
}

pub fn accept(frame: &mut Frame, constraint:u32, bail:i32) -> i32 {
    let cur = match frame.constraints {
        Some(ref constraints) => &constraints[constraint as usize],
        None => return bail as i32,
    };
    match cur {
        &Constraint::Scan {ref e, ref a, ref v, ref register_mask} => {
            // if we aren't solving for something this scan cares about, then we
            // automatically accept it.
            if !check_bits(*register_mask, frame.row.solving_for) {
                // println!("auto accept {:?} {:?}", cur, frame.row.solving_for);
               return 1;
            }
            let resolved_e = frame.resolve(e);
            let resolved_a = frame.resolve(a);
            let resolved_v = frame.resolve(v);
            let checked = frame.index.check(resolved_e, resolved_a, resolved_v);
            // println!("scan accept {:?} {:?}", cur, checked);
            match checked {
                true => 1,
                false => bail,
            }
        },
        &Constraint::Function {ref op, ref out, ref params} => {
            // println!("function accept {:?}", cur);
            1
        },
        _ => { 1 }
    }
}

pub fn get_rounds(frame: &mut Frame, constraint:u32, bail:i32) -> i32 {
    let cur = match frame.constraints {
        Some(ref constraints) => &constraints[constraint as usize],
        None => return bail as i32,
    };
    match cur {
        &Constraint::Scan {ref e, ref a, ref v, ref register_mask} => {
            let resolved_e = frame.resolve(e);
            let resolved_a = frame.resolve(a);
            let resolved_v = frame.resolve(v);
            frame.rounds.compute_output_rounds(frame.distinct.iter(resolved_e, resolved_a, resolved_v));
            1
        },
        _ => { panic!("Get rounds on non-scan") }
    }
    // println!("get rounds!");

}

pub fn output(frame: &mut Frame, constraint:u32, next:i32) -> i32 {
    let cur = match frame.constraints {
        Some(ref constraints) => &constraints[constraint as usize],
        None => return next,
    };
    match cur {
        &Constraint::Insert {ref e, ref a, ref v} => {
            let c = Change {
                e: frame.resolve(e),
                a: frame.resolve(a),
                v: frame.resolve(v),
                n: 0,
                round: 0,
                transaction: 0,
                count: 1,
            };
            frame.distinct.distinct(&c, &mut frame.rounds);
            // println!("insert {:?}", results);
        },
        _ => {}
    };
    next
}

//-------------------------------------------------------------------------
// Field
//-------------------------------------------------------------------------

#[derive(Debug)]
pub enum Field {
    Register(usize),
    Value(u32),
}

pub fn register(ix: usize) -> Field {
    Field::Register(ix)
}

//-------------------------------------------------------------------------
// Interner
//-------------------------------------------------------------------------

#[derive(Eq, PartialEq, Hash)]
pub enum Internable {
    String(String),
    Number(u32),
    Null,
}

pub struct Interner {
    id_to_value: HashMap<Internable, u32>,
    value_to_id: Vec<Internable>,
    next_id: u32,
}

impl Interner {
    pub fn new() -> Interner {
        Interner {id_to_value: HashMap::new(), value_to_id:vec![Internable::Null], next_id:1}
    }

    pub fn internable_to_id(&mut self, thing:Internable) -> u32 {
        match self.id_to_value.get(&thing) {
            Some(&id) => id,
            None => {
                let next = self.next_id;
                self.id_to_value.insert(thing, next);
                // @FIXME: trying to fix this gets me into borrow checker sadness
                // self.value_to_id.push(thing.copy());
                self.next_id += 1;
                next
            }
        }
    }

    pub fn string(&mut self, string:&str) -> Field {
        let thing = Internable::String(string.to_string());
        Field::Value(self.internable_to_id(thing))
    }

    pub fn string_id(&mut self, string:&str) -> u32 {
        let thing = Internable::String(string.to_string());
        self.internable_to_id(thing)
    }

    pub fn number(&mut self, num:f32) -> Field {
        let bitpattern = unsafe {
            transmute::<f32, u32>(num)
        };
        let thing = Internable::Number(bitpattern);
        Field::Value(self.internable_to_id(thing))
    }

    pub fn number_id(&mut self, num:f32) -> u32 {
        let bitpattern = unsafe {
            transmute::<f32, u32>(num)
        };
        let thing = Internable::Number(bitpattern);
        self.internable_to_id(thing)
    }
}

//-------------------------------------------------------------------------
// Constraint
//-------------------------------------------------------------------------

#[derive(Debug)]
pub enum Constraint {
    Scan {e: Field, a: Field, v: Field, register_mask: u64},
    Function {op: String, out: Vec<Field>, params: Vec<Field>},
    Insert {e: Field, a: Field, v:Field},
}

pub fn make_register_mask(fields: Vec<&Field>) -> u64 {
    let mut mask = 0;
    for field in fields {
        match field {
            &Field::Register(r) => mask = set_bit(mask, r as u32),
            _ => {},
        }
    }
    mask
}

pub fn make_scan(e:Field, a:Field, v:Field) -> Constraint {
    let register_mask = make_register_mask(vec![&e,&a,&v]);
    Constraint::Scan{e, a, v, register_mask }
}

//-------------------------------------------------------------------------
// Bit helpers
//-------------------------------------------------------------------------

fn check_bits(solved:u64, checking:u64) -> bool {
    solved & checking == checking
}

fn has_bit(solved:u64, bit:u64) -> bool {
    (solved >> bit) & 1 == 1
}

fn set_bit(solved:u64, bit:u32) -> u64 {
    solved | (1 << bit)
}

fn clear_bit(solved:u64, bit:u32) -> u64 {
    solved & !(1 << bit)
}

//-------------------------------------------------------------------------
// Interpret
//-------------------------------------------------------------------------

pub fn interpret(mut frame:&mut Frame, pipe:&Vec<Instruction>) {
    // println!("Doing work");
    let mut pointer:i32 = 0;
    let len = pipe.len() as i32;
    while pointer < len {
        let inst = &pipe[pointer as usize];
        pointer += match *inst {
            Instruction::start_block {block} => {
                start_block(&mut frame, block)
            },
            Instruction::get_iterator { iterator, constraint, bail } => {
                get_iterator(&mut frame, iterator, constraint, bail)
            },
            Instruction::iterator_next { iterator, bail } => {
                iterator_next(&mut frame, iterator, bail)
            },
            Instruction::accept { constraint, bail } => {
                accept(&mut frame, constraint, bail)
            },
            Instruction::get_rounds { constraint, bail } => {
                get_rounds(&mut frame, constraint, bail)
            },
            Instruction::output { constraint, next } => {
                output(&mut frame, constraint, next)
            },
            _ => {
                // println!("SKIPPING");
                1
            }
        }
    }
}

//-------------------------------------------------------------------------
// Round holder
//-------------------------------------------------------------------------

pub struct RoundHolder {
    output_rounds: Vec<(u32, i32)>,
    rounds: Vec<HashMap<(u32,u32,u32), Change>>,
    max_round: usize,
}

pub fn move_output_round(info:&Option<(u32, i32)>, round:&mut u32, count:&mut i32) {
    if let &Some((r, c)) = info {
        *round = r;
        *count += c;
    }
}

impl RoundHolder {
    pub fn new() -> RoundHolder {
        let mut rounds = vec![];
        for _ in 0..100 {
            rounds.push(HashMap::new());
        }
        RoundHolder { rounds, output_rounds:vec![], max_round: 0 }
    }

    pub fn compute_output_rounds(&mut self, mut right_iter: DistinctIter) {
        let mut neue = vec![];
        {
            // let len = self.output_rounds.len();
            let mut left_iter = self.output_rounds.drain(..);
            let mut left_round = 0;
            let mut left_count = 0;
            let mut right_round = 0;
            let mut right_count = 0;
            let mut left = left_iter.next();
            let mut right = right_iter.next();
            let mut next_left = left_iter.next();
            let mut next_right = right_iter.next();
            move_output_round(&left, &mut left_round, &mut left_count);
            move_output_round(&right, &mut right_round, &mut right_count);
            while left != None || right != None {
                if left_round == right_round {
                    if let Some((round, count)) = left {
                        let total = count * right_count;
                        if total != 0 {
                            neue.push((left_round, total));
                        }
                    }
                } else if left_round > right_round {
                    while next_right != None && next_right.unwrap().0 < left_round {
                        right = next_right;
                        next_right = right_iter.next();
                        move_output_round(&right, &mut right_round, &mut right_count);
                    }
                    if let Some((round, count)) = left {
                        let total = count * right_count;
                        if total != 0 {
                            neue.push((left_round, total));
                        }
                    }
                } else {
                    while next_left != None && next_left.unwrap().0 < right_round {
                        left = next_left;
                        next_left = left_iter.next();
                        move_output_round(&left, &mut left_round, &mut left_count);
                    }
                    if let Some((round, count)) = right {
                        let total = count * left_count;
                        if total != 0 {
                            neue.push((right_round, total));
                        }
                    }
                }

                match (next_left, next_right) {
                    (None, None) => { break; },
                    (None, Some(_)) => {
                        right = next_right;
                        next_right = right_iter.next();
                        move_output_round(&right, &mut right_round, &mut right_count);
                    },
                    (Some(_), None) => {
                        left = next_left;
                        next_left = left_iter.next();
                        move_output_round(&left, &mut left_round, &mut left_count);
                    },
                    (Some((next_left_count, _)), Some((next_right_count, _))) => {
                        if next_left_count <= next_right_count {
                            left = next_left;
                            next_left = left_iter.next();
                            move_output_round(&left, &mut left_round, &mut left_count);
                        } else {
                            right = next_right;
                            next_right = right_iter.next();
                            move_output_round(&right, &mut right_round, &mut right_count);
                        }
                    }
                }

            }
        }
        self.output_rounds = neue;
    }

    pub fn output(&self, distinct:&mut DistinctIndex, e:u32, a:u32, v:u32, n:u32) {

    }

    pub fn insert(&mut self, change:Change) {
        let key = (change.e, change.a, change.v);
        let round = change.round as usize;
        self.max_round = cmp::max(round, self.max_round);
        match self.rounds[round].entry(key) {
            Entry::Occupied(mut o) => {
                o.get_mut().count += change.count;
            }
            Entry::Vacant(o) => {
                o.insert(change);
            }
        };
    }

    pub fn clear(&mut self) {
        for ix in 0..self.max_round {
            self.rounds[ix].clear();
        }
        self.max_round = 0;
    }

    pub fn iter(&self) -> RoundHolderIter {
        RoundHolderIter::new(self)
    }
}

pub struct RoundHolderIter<'a> {
    holder: &'a RoundHolder,
    max_round: usize,
    round_ix: usize,
    change_ix: usize,
    cur_changes: Vec<&'a Change>,
}

impl<'a> RoundHolderIter<'a> {
    pub fn new(holder: &'a RoundHolder) -> RoundHolderIter<'a> {
        RoundHolderIter { holder, round_ix: 0, change_ix: 0, max_round: holder.max_round, cur_changes: vec![] }
    }
}

impl<'a> Iterator for RoundHolderIter<'a> {
    type Item = Change;

    fn next(&mut self) -> Option<Change> {
        let ref mut cur_changes = self.cur_changes;
        let mut round_ix = self.round_ix;
        let mut change_ix = self.change_ix;
        let mut max_round = self.max_round;
        if change_ix >= cur_changes.len() {
            cur_changes.clear();
            change_ix = 0;
            while round_ix <= max_round + 1 && cur_changes.len() == 0 {
                for change in self.holder.rounds[round_ix].values() {
                    cur_changes.push(change);
                }
                round_ix += 1;
            }
        }
        self.change_ix = change_ix + 1;
        self.round_ix = round_ix;
        match cur_changes.get(change_ix) {
            None => None,
            Some(&change) => Some(change.clone()),
        }
    }
}


//-------------------------------------------------------------------------
// Tests
//-------------------------------------------------------------------------

pub fn doit() {
    // prog.block("simple block", ({find, record, lib}) => {
    //  let person = find("person");
    //  let text = `name: ${person.name}`;
    //  return [
    //    record("html/div", {person, text})
    //  ]
    // });
    //
    let mut int = Interner::new();
    let constraints = vec![
        make_scan(register(0), int.string("tag"), int.string("person")),
        make_scan(register(0), int.string("name"), register(1)),
        Constraint::Function {op: "concat".to_string(), out: vec![register(2)], params: vec![int.string("name: "), register(1)]},
        Constraint::Function {op: "gen_id".to_string(), out: vec![register(3)], params: vec![register(0), register(2)]},
        // Constraint::Insert {e: register(3), a: int.string("tag"), v: int.string("html/div")},
        // Constraint::Insert {e: register(3), a: int.string("person"), v: register(0)},
        // Constraint::Insert {e: register(3), a: int.string("text"), v: register(2)},
        Constraint::Insert {e: int.string("foo"), a: int.string("tag"), v: int.string("html/div")},
        Constraint::Insert {e: int.string("foo"), a: int.string("person"), v: register(0)},
        Constraint::Insert {e: int.string("foo"), a: int.string("text"), v: register(1)},
    ];

    let blocks = vec![
        Block { name: "simple block".to_string(), constraints },
    ];

    let pipe = vec![
        Instruction::start_block { block:0 },

        Instruction::get_iterator {bail: 100000, constraint: 0, iterator: 0},
        Instruction::get_iterator {bail: 100000, constraint: 1, iterator: 0},
        // Instruction::get_iterator {bail: 100000, constraint: 2, iterator: 0},
        // Instruction::get_iterator {bail: 100000, constraint: 3, iterator: 0},

        Instruction::iterator_next { bail: 100000, iterator: 0},

        Instruction::accept {bail: -1, constraint: 0},
        Instruction::accept {bail: -2, constraint: 1},
        Instruction::accept {bail: -3, constraint: 2},
        Instruction::accept {bail: -4, constraint: 3},

        Instruction::get_iterator {bail: -5, constraint: 0, iterator: 1},
        Instruction::get_iterator {bail: -6, constraint: 1, iterator: 1},
        // Instruction::get_iterator {bail: -7, constraint: 2, iterator: 1},
        // Instruction::get_iterator {bail: -8, constraint: 3, iterator: 1},

        Instruction::iterator_next { bail: -7, iterator: 1},

        Instruction::accept {bail: -1, constraint: 0},
        Instruction::accept {bail: -2, constraint: 1},
        Instruction::accept {bail: -3, constraint: 2},
        Instruction::accept {bail: -4, constraint: 3},

        // Instruction::get_iterator {bail: -5, constraint: 0, iterator: 2},
        // Instruction::get_iterator {bail: -6, constraint: 1, iterator: 2},
        // Instruction::get_iterator {bail: -7, constraint: 2, iterator: 2},
        // Instruction::get_iterator {bail: -8, constraint: 3, iterator: 2},

        // Instruction::iterator_next { bail: -9, iterator: 2 },

        // Instruction::accept {bail: -1, constraint: 0},
        // Instruction::accept {bail: -2, constraint: 1},
        // Instruction::accept {bail: -3, constraint: 2},
        // Instruction::accept {bail: -4, constraint: 3},

        // Instruction::get_iterator {bail: -5, constraint: 0, iterator:3},
        // Instruction::get_iterator {bail: -6, constraint: 1, iterator:3},
        // Instruction::get_iterator {bail: -7, constraint: 2, iterator:3},
        // Instruction::get_iterator {bail: -8, constraint: 3, iterator:3},

        // Instruction::iterator_next { bail: -9, iterator: 3 },

        // Instruction::accept {bail: -1, constraint: 0},
        // Instruction::accept {bail: -2, constraint: 1},
        // Instruction::accept {bail: -3, constraint: 2},
        // Instruction::accept {bail: -4, constraint: 3},

        Instruction::get_rounds {bail: -5, constraint: 0},
        Instruction::get_rounds {bail: -6, constraint: 1},

        Instruction::output {next: 1, constraint: 4},
        Instruction::output {next: 1, constraint: 5},
        Instruction::output {next: -9, constraint: 6},
    ];


    let change = Change { e:0, a:0, v:0, n:0, round:0, transaction:0, count:0};
    let mut distinct = DistinctIndex::new();
    let mut rounds = RoundHolder::new();
    let mut index = HashIndex::new();
    index.insert(int.string_id("foo"), int.string_id("tag"), int.string_id("person"));
    index.insert(int.string_id("foo"), int.string_id("name"), int.string_id("chris"));
    index.insert(int.string_id("meep"), int.string_id("tag"), int.string_id("person"));
    index.insert(int.string_id("meep"), int.string_id("name"), int.string_id("chris"));
    index.insert(int.string_id("joe"), int.string_id("tag"), int.string_id("person"));
    index.insert(int.string_id("eep"), int.string_id("name"), int.string_id("loop"));
    index.insert(int.string_id("eep2"), int.string_id("name"), int.string_id("loop"));
    // let start = Instant::now();
    for _ in 0..1 {
        let mut frame = Frame::new(&mut index, &mut rounds, &mut distinct, &blocks, &change);
        interpret(&mut frame, &pipe);
    }
    // println!("TOOK {:?}", start.elapsed());
}


// #[cfg(test)]
pub mod tests {
    extern crate test;

    use super::*;
    use self::test::Bencher;

    #[test]
    fn test_check_bits() {
        let solved = 45;
        let checking = 41;
        assert!(check_bits(solved, checking));
    }

    #[test]
    fn test_set_bit() {
        let mut solved = 41;
        let setting = 2;
        solved = set_bit(solved, setting);
        assert_eq!(45, solved);
    }

    #[test]
    fn test_has_bit() {
        let solved = 41;
        assert!(has_bit(solved, 5));
        assert!(has_bit(solved, 3));
        assert!(has_bit(solved, 0));
        assert!(!has_bit(solved, 1));
        assert!(!has_bit(solved, 2));
    }

    fn check_output_rounds(existing: Vec<(u32, i32)>, neueRounds: Vec<i32>, expected: Vec<(u32, i32)>) {
        let mut holder = RoundHolder::new();
        let mut iter = DistinctIter::new(&neueRounds);
        holder.output_rounds = existing;
        holder.compute_output_rounds(iter);
        assert_eq!(holder.output_rounds, expected);

    }

    #[test]
    fn round_holder_compute_output_rounds() {
        check_output_rounds(vec![(3,1), (5,1)], vec![1,-1,0,0,1,0,-1], vec![(4,1), (5,1), (6,-2)]);
        check_output_rounds(vec![(3,1), (5,1)], vec![1,-1,0,1,0,0,-1], vec![(3,1), (5,1), (6,-2)]);
        check_output_rounds(vec![(3,1), (5,1)], vec![1,-1,0,0], vec![]);
        check_output_rounds(vec![(3,1), (5,1)], vec![1,0,0,0,0,0,-1], vec![(3,1), (5,1), (6,-2)]);
        check_output_rounds(vec![(0,1), (6,-1)], vec![1,0,0,0,0,0,-1], vec![(0,1), (6,-1)]);
    }

    #[bench]
    pub fn round_holder_compute_output_rounds_bench(b:&mut Bencher) {
        let mut holder = RoundHolder::new();
        let rounds = vec![1,-1,0,0,1,0,-1];
        holder.output_rounds = vec![(3,1), (5,1)];
        b.iter(|| {
            let mut iter = DistinctIter::new(&rounds);
            holder.compute_output_rounds(iter);
        });
    }

    // #[test]
    pub fn test_simple_GJ() {
        // prog.block("simple block", ({find, record, lib}) => {
        //  let person = find("person");
        //  let text = `name: ${person.name}`;
        //  return [
        //    record("html/div", {person, text})
        //  ]
        // });
        //
        let mut int = Interner::new();
        let constraints = vec![
            make_scan(register(0), int.string("tag"), int.string("person")),
            make_scan(register(0), int.string("name"), register(1)),
            Constraint::Function {op: "concat".to_string(), out: vec![register(2)], params: vec![int.string("name: "), register(1)]},
            Constraint::Function {op: "gen_id".to_string(), out: vec![register(3)], params: vec![register(0), register(2)]},
            // Constraint::Insert {e: register(3), a: int.string("tag"), v: int.string("html/div")},
            // Constraint::Insert {e: register(3), a: int.string("person"), v: register(0)},
            // Constraint::Insert {e: register(3), a: int.string("text"), v: register(2)},
            Constraint::Insert {e: int.string("foo"), a: int.string("tag"), v: int.string("html/div")},
            Constraint::Insert {e: int.string("foo"), a: int.string("person"), v: register(0)},
            Constraint::Insert {e: int.string("foo"), a: int.string("text"), v: register(1)},
        ];

        let blocks = vec![
            Block { name: "simple block".to_string(), constraints },
        ];

        let pipe = vec![
            Instruction::start_block { block:0 },

            Instruction::get_iterator {bail: 100000, constraint: 0, iterator: 0},
            Instruction::get_iterator {bail: 100000, constraint: 1, iterator: 0},
            // Instruction::get_iterator {bail: 100000, constraint: 2, iterator: 0},
            // Instruction::get_iterator {bail: 100000, constraint: 3, iterator: 0},

            Instruction::iterator_next { bail: 100000, iterator: 0},

            Instruction::accept {bail: -1, constraint: 0},
            Instruction::accept {bail: -2, constraint: 1},
            Instruction::accept {bail: -3, constraint: 2},
            Instruction::accept {bail: -4, constraint: 3},

            Instruction::get_iterator {bail: -5, constraint: 0, iterator: 1},
            Instruction::get_iterator {bail: -6, constraint: 1, iterator: 1},
            // Instruction::get_iterator {bail: -7, constraint: 2, iterator: 1},
            // Instruction::get_iterator {bail: -8, constraint: 3, iterator: 1},

            Instruction::iterator_next { bail: -7, iterator: 1},

            Instruction::accept {bail: -1, constraint: 0},
            Instruction::accept {bail: -2, constraint: 1},
            Instruction::accept {bail: -3, constraint: 2},
            Instruction::accept {bail: -4, constraint: 3},

            // Instruction::get_iterator {bail: -5, constraint: 0, iterator: 2},
            // Instruction::get_iterator {bail: -6, constraint: 1, iterator: 2},
            // Instruction::get_iterator {bail: -7, constraint: 2, iterator: 2},
            // Instruction::get_iterator {bail: -8, constraint: 3, iterator: 2},

            // Instruction::iterator_next { bail: -9, iterator: 2 },

            // Instruction::accept {bail: -1, constraint: 0},
            // Instruction::accept {bail: -2, constraint: 1},
            // Instruction::accept {bail: -3, constraint: 2},
            // Instruction::accept {bail: -4, constraint: 3},

            // Instruction::get_iterator {bail: -5, constraint: 0, iterator:3},
            // Instruction::get_iterator {bail: -6, constraint: 1, iterator:3},
            // Instruction::get_iterator {bail: -7, constraint: 2, iterator:3},
            // Instruction::get_iterator {bail: -8, constraint: 3, iterator:3},

            // Instruction::iterator_next { bail: -9, iterator: 3 },

            // Instruction::accept {bail: -1, constraint: 0},
            // Instruction::accept {bail: -2, constraint: 1},
            // Instruction::accept {bail: -3, constraint: 2},
            // Instruction::accept {bail: -4, constraint: 3},

            Instruction::get_rounds {bail: -5, constraint: 0},
            Instruction::get_rounds {bail: -6, constraint: 1},

            Instruction::output {next: 1, constraint: 4},
            Instruction::output {next: 1, constraint: 5},
            Instruction::output {next: -9, constraint: 6},
        ];

        let change = Change { e:0, a:0, v:0, n:0, round:0, transaction:0, count:0};
        let mut distinct = DistinctIndex::new();
        let mut rounds = RoundHolder::new();
        let mut index = HashIndex::new();
        index.insert(int.string_id("foo"), int.string_id("tag"), int.string_id("person"));
        index.insert(int.string_id("foo"), int.string_id("name"), int.string_id("chris"));
        index.insert(int.string_id("meep"), int.string_id("tag"), int.string_id("person"));
        index.insert(int.string_id("meep"), int.string_id("name"), int.string_id("chris"));
        index.insert(int.string_id("joe"), int.string_id("tag"), int.string_id("person"));
        index.insert(int.string_id("eep"), int.string_id("name"), int.string_id("loop"));
        index.insert(int.string_id("eep2"), int.string_id("name"), int.string_id("loop"));

        // let start = Instant::now();
        let mut frame = Frame::new(&mut index, &mut rounds, &mut distinct, &blocks, &change);
        interpret(&mut frame, &pipe);
        // println!("TOOK {:?}", start.elapsed());
    }


    #[bench]
    fn bench_simple_GJ(b:&mut Bencher) {
        // prog.block("simple block", ({find, record, lib}) => {
        //  let person = find("person");
        //  let text = `name: ${person.name}`;
        //  return [
        //    record("html/div", {person, text})
        //  ]
        // });
        //
        let mut int = Interner::new();
        let constraints = vec![
            make_scan(register(0), int.string("tag"), int.string("person")),
            make_scan(register(0), int.string("name"), register(1)),
            Constraint::Function {op: "concat".to_string(), out: vec![register(2)], params: vec![int.string("name: "), register(1)]},
            Constraint::Function {op: "gen_id".to_string(), out: vec![register(3)], params: vec![register(0), register(2)]},
            // Constraint::Insert {e: register(3), a: int.string("tag"), v: int.string("html/div")},
            // Constraint::Insert {e: register(3), a: int.string("person"), v: register(0)},
            // Constraint::Insert {e: register(3), a: int.string("text"), v: register(2)},
            Constraint::Insert {e: int.string("foo"), a: int.string("tag"), v: int.string("html/div")},
            Constraint::Insert {e: int.string("foo"), a: int.string("person"), v: register(0)},
            Constraint::Insert {e: int.string("foo"), a: int.string("text"), v: register(1)},
        ];

        let blocks = vec![
            Block { name: "simple block".to_string(), constraints },
        ];

        let pipe = vec![
            Instruction::start_block { block:0 },

            Instruction::get_iterator {bail: 100000, constraint: 0, iterator: 0},
            Instruction::get_iterator {bail: 100000, constraint: 1, iterator: 0},
            // Instruction::get_iterator {bail: 100000, constraint: 2, iterator: 0},
            // Instruction::get_iterator {bail: 100000, constraint: 3, iterator: 0},

            Instruction::iterator_next { bail: 100000, iterator: 0},

            Instruction::accept {bail: -1, constraint: 0},
            Instruction::accept {bail: -2, constraint: 1},
            Instruction::accept {bail: -3, constraint: 2},
            Instruction::accept {bail: -4, constraint: 3},

            Instruction::get_iterator {bail: -5, constraint: 0, iterator: 1},
            Instruction::get_iterator {bail: -6, constraint: 1, iterator: 1},
            // Instruction::get_iterator {bail: -7, constraint: 2, iterator: 1},
            // Instruction::get_iterator {bail: -8, constraint: 3, iterator: 1},

            Instruction::iterator_next { bail: -7, iterator: 1},

            Instruction::accept {bail: -1, constraint: 0},
            Instruction::accept {bail: -2, constraint: 1},
            Instruction::accept {bail: -3, constraint: 2},
            Instruction::accept {bail: -4, constraint: 3},

            // Instruction::get_iterator {bail: -5, constraint: 0, iterator: 2},
            // Instruction::get_iterator {bail: -6, constraint: 1, iterator: 2},
            // Instruction::get_iterator {bail: -7, constraint: 2, iterator: 2},
            // Instruction::get_iterator {bail: -8, constraint: 3, iterator: 2},

            // Instruction::iterator_next { bail: -9, iterator: 2 },

            // Instruction::accept {bail: -1, constraint: 0},
            // Instruction::accept {bail: -2, constraint: 1},
            // Instruction::accept {bail: -3, constraint: 2},
            // Instruction::accept {bail: -4, constraint: 3},

            // Instruction::get_iterator {bail: -5, constraint: 0, iterator:3},
            // Instruction::get_iterator {bail: -6, constraint: 1, iterator:3},
            // Instruction::get_iterator {bail: -7, constraint: 2, iterator:3},
            // Instruction::get_iterator {bail: -8, constraint: 3, iterator:3},

            // Instruction::iterator_next { bail: -9, iterator: 3 },

            // Instruction::accept {bail: -1, constraint: 0},
            // Instruction::accept {bail: -2, constraint: 1},
            // Instruction::accept {bail: -3, constraint: 2},
            // Instruction::accept {bail: -4, constraint: 3},

            Instruction::get_rounds {bail: -5, constraint: 0},
            Instruction::get_rounds {bail: -6, constraint: 1},

            Instruction::output {next: 1, constraint: 4},
            Instruction::output {next: 1, constraint: 5},
            Instruction::output {next: -9, constraint: 6},
        ];

        let mut distinct = DistinctIndex::new();
        let mut rounds = RoundHolder::new();
        let mut index = HashIndex::new();
        index.insert(int.string_id("foo"), int.string_id("tag"), int.string_id("person"));
        index.insert(int.string_id("foo"), int.string_id("name"), int.string_id("chris"));
        index.insert(int.string_id("meep"), int.string_id("tag"), int.string_id("person"));
        index.insert(int.string_id("meep"), int.string_id("name"), int.string_id("chris"));
        index.insert(int.string_id("joe"), int.string_id("tag"), int.string_id("person"));
        index.insert(int.string_id("eep"), int.string_id("name"), int.string_id("loop"));
        index.insert(int.string_id("eep2"), int.string_id("name"), int.string_id("loop"));

        b.iter(|| {
            let change = Change { e:0, a:0, v:0, n:0, round:0, transaction:0, count:0};
            let mut frame = Frame::new(&mut index, &mut rounds, &mut distinct, &blocks, &change);
            interpret(&mut frame, &pipe);
        })
    }

}

