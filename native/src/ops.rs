//-------------------------------------------------------------------------
// Ops
//-------------------------------------------------------------------------

use indexes::BitIndex;
use std::collections::HashMap;
use std::mem::transmute;

//-------------------------------------------------------------------------
// Frame
//-------------------------------------------------------------------------

pub struct Change {
    e: u32,
    a: u32,
    v: u32,
    n: u32,
    round: u32,
    transaction: u32,
    count: u32,
}

//-------------------------------------------------------------------------
// Block
//-------------------------------------------------------------------------

pub struct Block {
    name: String,
    constraints: Vec<Constraint>,
}

//-------------------------------------------------------------------------
// Frame
//-------------------------------------------------------------------------

pub struct Frame<'a> {
    input: &'a Change,
    vars: Vec<u32>,
    solved_vars: u64,
    index: &'a BitIndex,
    constraints: Option<&'a Vec<Constraint>>,
    blocks: &'a Vec<Block>,
}

impl<'a> Frame<'a> {
    pub fn new(index: &'a BitIndex, blocks: &'a Vec<Block>, input:&'a Change) -> Frame<'a> {
        Frame {vars: vec![], solved_vars:0, index, input, blocks, constraints: None}
    }
}



//-------------------------------------------------------------------------
// Instruction
//-------------------------------------------------------------------------

pub enum Instruction {
    start_block { block: u32 },
    get_iterator {bail: i32, constraint: u32},
    iterator_next {bail: i32},
    accept {bail: i32, constraint:u32},
    output {next: i32, constraint:u32},
}

pub fn start_block(frame: &mut Frame, block:u32) -> usize {
    println!("STARTING! {:?}", block);
    frame.constraints = Some(&frame.blocks[block as usize].constraints);
    1
}

pub fn get_iterator(frame: &mut Frame, constraint:u32, bail:i32) -> usize {
    let cur = match frame.constraints {
        Some(ref constraints) => &constraints[constraint as usize],
        None => return bail as usize,
    };
    match *cur {
        Constraint::Scan {ref e, ref a, ref v} => {
            println!("get scan iterator {:?}", cur);
        },
        Constraint::Function {ref op, ref out, ref params} => {
            println!("get function iterator {:?}", cur);
        },
        _ => {}
    };
    1
}

pub fn iterator_next(frame: &mut Frame, bail:i32) -> usize {
    println!("next!");
    1
}

pub fn accept(frame: &mut Frame, constraint:u32, bail:i32) -> usize {
    println!("accept! {:?}", constraint);
    1
}

pub fn output(frame: &mut Frame, constraint:u32, next:i32) -> usize {
    println!("output! {:?}", constraint);
    1
}

//-------------------------------------------------------------------------
// Field
//-------------------------------------------------------------------------

#[derive(Debug)]
pub enum Field {
    Register(u32),
    Value(u32),
}

pub fn register(ix: u32) -> Field {
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
                self.value_to_id.push(thing);
                self.next_id += 1;
                next
            }
        }
    }

    pub fn string(&mut self, string:&str) -> Field {
        let thing = Internable::String(string.to_string());
        Field::Value(self.internable_to_id(thing))
    }

    pub fn number(&mut self, num:f32) -> Field {
        let bitpattern = unsafe {
            transmute::<f32, u32>(num)
        };
        let thing = Internable::Number(bitpattern);
        Field::Value(self.internable_to_id(thing))
    }
}

//-------------------------------------------------------------------------
// Scan
//-------------------------------------------------------------------------

pub struct Scan {
    e: Field,
    a: Field,
    v: Field,
}

//-------------------------------------------------------------------------
// Function
//-------------------------------------------------------------------------

pub struct Function {
    op: String,
    out: Vec<Field>,
    params: Vec<Field>,
}

//-------------------------------------------------------------------------
// Insert
//-------------------------------------------------------------------------

pub struct Insert {
    e: Field,
    a: Field,
    v: Field,
}

//-------------------------------------------------------------------------
// Insert
//-------------------------------------------------------------------------

#[derive(Debug)]
pub enum Constraint {
    Scan {e: Field, a: Field, v: Field},
    Function {op: String, out: Vec<Field>, params: Vec<Field>},
    Insert {e: Field, a: Field, v:Field},
}

//-------------------------------------------------------------------------
// Bit helpers
//-------------------------------------------------------------------------

fn check_bits(solved:u64, checking:u64) -> bool {
    solved & checking == checking
}

fn has_bit(solved:u64, bit:u32) -> bool {
    (solved >> bit) & 1 == 1
}

fn set_bit(solved:&mut u64, bit:u32) {
    *solved |= 1 << bit
}

//-------------------------------------------------------------------------
// Interpret
//-------------------------------------------------------------------------

pub fn interpret(pipe:&Vec<Instruction>, blocks:&Vec<Block>) {
    println!("Doing work");
    let change = Change { e:0, a:0, v:0, n:0, round:0, transaction:0, count:0};
    let mut index = BitIndex::new();
    let mut frame = Frame::new(&index, blocks, &change);
    let mut pointer = 0;
    let len = pipe.len();
    while pointer < len {
        let inst = &pipe[pointer];
        pointer += match *inst {
            Instruction::start_block {block} => {
                start_block(&mut frame, block)
            },
            Instruction::get_iterator { constraint, bail } => {
                get_iterator(&mut frame, constraint, bail)
            },
            Instruction::iterator_next { bail } => {
                iterator_next(&mut frame, bail)
            },
            Instruction::accept { constraint, bail } => {
                accept(&mut frame, constraint, bail)
            },
            Instruction::output { constraint, next } => {
                output(&mut frame, constraint, next)
            },
            _ => {
                println!("SKIPPING");
                1
            }
        }
    }
}

//-------------------------------------------------------------------------
// Tests
//-------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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
        set_bit(&mut solved, setting);
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

    #[test]
    fn test_simple_GJ() {
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
            Constraint::Scan {e: register(0), a: int.string("tag"), v: int.string("person")},
            Constraint::Scan {e: register(0), a: int.string("name"), v: register(1)},
            Constraint::Function {op: "concat".to_string(), out: vec![register(2)], params: vec![int.string("name: "), register(1)]},
            Constraint::Function {op: "gen_id".to_string(), out: vec![register(3)], params: vec![register(0), register(2)]},
            Constraint::Insert {e: register(3), a: int.string("tag"), v: int.string("html/div")},
            Constraint::Insert {e: register(3), a: int.string("person"), v: register(0)},
            Constraint::Insert {e: register(3), a: int.string("text"), v: register(2)},
        ];

        let blocks = vec![
            Block { name: "simple block".to_string(), constraints },
        ];

        let pipe = vec![
            Instruction::start_block { block:0 },

            Instruction::get_iterator {bail: 0, constraint: 0},
            Instruction::get_iterator {bail: 0, constraint: 1},
            Instruction::get_iterator {bail: 0, constraint: 2},
            Instruction::get_iterator {bail: 0, constraint: 3},

            Instruction::iterator_next { bail: 0 },

            Instruction::accept {bail: -1, constraint: 0},
            Instruction::accept {bail: -2, constraint: 1},
            Instruction::accept {bail: -3, constraint: 2},
            Instruction::accept {bail: -4, constraint: 3},

            Instruction::get_iterator {bail: -5, constraint: 0},
            Instruction::get_iterator {bail: -6, constraint: 1},
            Instruction::get_iterator {bail: -7, constraint: 2},
            Instruction::get_iterator {bail: -8, constraint: 3},

            Instruction::iterator_next { bail: -9 },

            Instruction::accept {bail: -1, constraint: 0},
            Instruction::accept {bail: -2, constraint: 1},
            Instruction::accept {bail: -3, constraint: 2},
            Instruction::accept {bail: -4, constraint: 3},

            Instruction::get_iterator {bail: -5, constraint: 0},
            Instruction::get_iterator {bail: -6, constraint: 1},
            Instruction::get_iterator {bail: -7, constraint: 2},
            Instruction::get_iterator {bail: -8, constraint: 3},

            Instruction::iterator_next { bail: -9 },

            Instruction::accept {bail: -1, constraint: 0},
            Instruction::accept {bail: -2, constraint: 1},
            Instruction::accept {bail: -3, constraint: 2},
            Instruction::accept {bail: -4, constraint: 3},

            Instruction::get_iterator {bail: -5, constraint: 0},
            Instruction::get_iterator {bail: -6, constraint: 1},
            Instruction::get_iterator {bail: -7, constraint: 2},
            Instruction::get_iterator {bail: -8, constraint: 3},

            Instruction::iterator_next { bail: -9 },

            Instruction::accept {bail: -1, constraint: 0},
            Instruction::accept {bail: -2, constraint: 1},
            Instruction::accept {bail: -3, constraint: 2},
            Instruction::accept {bail: -4, constraint: 3},

            Instruction::output {next: 1, constraint: 4},
            Instruction::output {next: -6, constraint: 5},
        ];

        interpret(&pipe, &blocks);
    }

}

