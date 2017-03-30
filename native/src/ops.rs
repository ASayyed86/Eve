
fn check_bits(solved:u64, checking:u64) -> bool {
    solved & checking == checking
}

fn has_bit(solved:u64, bit:u32) -> bool {
    (solved >> bit) & 1 == 1
}

fn set_bit(solved:&mut u64, bit:u32) {
    *solved |= 1 << bit
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

