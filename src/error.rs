use std::fmt;

#[derive(Debug)]
pub enum DelaunayError {
    TooFewPoints(usize),
    Degenerate(String),
    Corrupt(String),
}

impl fmt::Display for DelaunayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DelaunayError::TooFewPoints(n) => {
                write!(f, "not enough distinct points (got {})", n)
            }
            DelaunayError::Degenerate(msg) => write!(f, "degenerate input: {}", msg),
            DelaunayError::Corrupt(msg) => write!(f, "internal corruption: {}", msg),
        }
    }
}
