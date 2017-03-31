//-------------------------------------------------------------------------
// Ops
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
// Frame
//-------------------------------------------------------------------------

struct Frame {
    vars: Vec<u32>,
    solved_vars: u64,
}

//-------------------------------------------------------------------------
// Codes
//-------------------------------------------------------------------------

enum ByteCodes {
    get_iterator,
    iterator_next,
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

pub struct Instruction {
    func: fn(i32, i32) -> i32,
    b:i32,
}

pub fn adder(a:i32, b:i32) -> i32 {
    a + b
}
pub fn subber(a:i32, b:i32) -> i32 {
    a - b
}


pub fn interpret(foo:&Vec<Instruction>) -> i32 {
    let mut x = 0;
    for inst in foo.iter() {
        x = (inst.func)(x, inst.b);
    }
    x
}

pub enum OP {
    add,
    remove,
}

pub struct Inst2 {
    op: OP,
    v: i32,
}

pub fn interpret2(foo:&Vec<Inst2>) -> i32 {
    let mut x = 0;
    for inst in foo.iter() {
        x = match inst.op {
            OP::add => adder(x, inst.v),
            OP::remove => subber(x, inst.v),
        }
    }
    x
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

    // #[bench]
    // fn benchmark(b:&mut Bencher) {
    //     let mut c = 0;
    //     let mut instructions = vec![];
    //     for ix in 0..500000 {
    //         instructions.push(Instruction {func: adder, b: ix % 5});
    //         instructions.push(Instruction {func: subber, b: ix % 3});
    //     }
    //     b.iter(|| {
    //         c = interpret(&instructions);
    //     });
    // }

    #[bench]
    fn benchmark2(b:&mut Bencher) {
        let mut c = 0;
        let mut instructions = vec![];
        for ix in 0..5000000 {
            instructions.push(Inst2 {op: OP::add, v: ix % 5});
            instructions.push(Inst2 {op: OP::remove, v: ix % 8});
        }
        b.iter(|| {
            c = interpret2(&instructions);
        });
        println!("{:?}", c)
    }
}

