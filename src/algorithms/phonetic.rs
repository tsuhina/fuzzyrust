//! Phonetic matching algorithms
//!
//! These encode strings by how they sound, useful for name matching
//! and handling spelling variations of the same pronunciation.
//!
//! # Algorithms
//! - **Soundex**: Classic 4-character code (first letter + 3 digits)
//! - **Metaphone**: More accurate phonetic encoding with variable length

use super::Similarity;

/// Soundex phonetic encoder
///
/// Produces a 4-character code: first letter + 3 digits.
/// Stateless encoder - all instances are equivalent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Soundex;

impl Soundex {
    pub fn new() -> Self {
        Self
    }
    
    /// Encode a string to its Soundex code
    pub fn encode(&self, s: &str) -> String {
        soundex(s)
    }
}

impl Similarity for Soundex {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        soundex_similarity(a, b)
    }
    
    fn name(&self) -> &'static str {
        "soundex"
    }
}

/// Metaphone phonetic encoder (more accurate than Soundex)
///
/// # Parameters
/// - `max_length`: Maximum code length (default: 4, increase for longer words)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metaphone {
    /// Maximum code length
    pub max_length: usize,
}

impl Default for Metaphone {
    fn default() -> Self {
        Self { max_length: 4 }
    }
}

impl Metaphone {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_max_length(max_length: usize) -> Self {
        Self { max_length }
    }
    
    /// Encode a string to its Metaphone code
    pub fn encode(&self, s: &str) -> String {
        metaphone(s, self.max_length)
    }
}

impl Similarity for Metaphone {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        metaphone_similarity(a, b, self.max_length)
    }
    
    fn name(&self) -> &'static str {
        "metaphone"
    }
}

/// Encode a string using the Soundex algorithm.
/// Returns a 4-character code: first letter + 3 digits.
pub fn soundex(s: &str) -> String {
    let s = s.trim().to_uppercase();
    
    if s.is_empty() {
        return String::new();
    }
    
    let chars: Vec<char> = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    
    if chars.is_empty() {
        return String::new();
    }
    
    let first = chars[0];
    
    let encode_char = |c: char| -> char {
        match c {
            'B' | 'F' | 'P' | 'V' => '1',
            'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => '2',
            'D' | 'T' => '3',
            'L' => '4',
            'M' | 'N' => '5',
            'R' => '6',
            _ => '0', // A, E, I, O, U, H, W, Y
        }
    };
    
    // Helper to check if a character is H or W (ignored entirely in Soundex)
    let is_hw = |c: char| -> bool {
        matches!(c, 'H' | 'W')
    };

    let mut result = String::with_capacity(4);
    result.push(first);

    let mut prev_code = encode_char(first);

    for &c in &chars[1..] {
        if result.len() >= 4 {
            break;
        }

        let code = encode_char(c);

        // H and W are completely ignored (don't affect prev_code)
        if is_hw(c) {
            continue;
        }

        // Skip if same as previous or if it's a vowel (code 0)
        if code != '0' && code != prev_code {
            result.push(code);
        }

        // Update prev_code for adjacency detection
        // Vowels (code 0) break adjacency, H/W don't (handled above)
        prev_code = code;
    }

    // Pad with zeros if needed
    while result.len() < 4 {
        result.push('0');
    }

    result
}

/// Check if two strings have the same Soundex code.
pub fn soundex_match(a: &str, b: &str) -> bool {
    soundex(a) == soundex(b)
}

/// Soundex similarity: 1.0 if codes match, 0.0 otherwise.
/// For partial matching, compare individual code positions.
pub fn soundex_similarity(a: &str, b: &str) -> f64 {
    let code_a = soundex(a);
    let code_b = soundex(b);
    
    if code_a.is_empty() || code_b.is_empty() {
        return 0.0;
    }
    
    if code_a == code_b {
        return 1.0;
    }
    
    // Partial matching: count matching positions
    let matches: usize = code_a.chars()
        .zip(code_b.chars())
        .filter(|(a, b)| a == b)
        .count();
    
    matches as f64 / 4.0
}

/// Encode a string using the Metaphone algorithm.
///
/// Note: TH is encoded as '0' representing the theta sound. This is a common
/// convention in phonetic algorithms where '0' serves as a placeholder for
/// sounds that don't map directly to consonant codes.
pub fn metaphone(s: &str, max_length: usize) -> String {
    let s = s.trim().to_uppercase();
    let chars: Vec<char> = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    
    if chars.is_empty() {
        return String::new();
    }
    
    let mut result = String::with_capacity(max_length);
    let len = chars.len();
    let mut i = 0;
    
    // Handle initial letter combinations
    if len >= 2 {
        match (chars[0], chars[1]) {
            ('K', 'N') | ('G', 'N') | ('P', 'N') | ('A', 'E') | ('W', 'R') => {
                i = 1;
            }
            ('W', 'H') => {
                result.push('W');
                i = 2;
            }
            _ => {}
        }
    }
    
    if i == 0 && chars[0] == 'X' {
        result.push('S');
        i = 1;
    }
    
    while i < len && result.len() < max_length {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let prev = if i > 0 { chars.get(i - 1).copied() } else { None };
        let next2 = chars.get(i + 2).copied();
        
        match c {
            'A' | 'E' | 'I' | 'O' | 'U' => {
                if i == 0 {
                    result.push(c);
                }
            }
            'B' => {
                // B is silent at end of word after M (e.g., "dumb", "lamb", "climb")
                // Add B unless: preceded by M AND at end of word
                if prev != Some('M') || i != len - 1 {
                    result.push('B');
                }
            }
            'C' => {
                if next == Some('I') && next2 == Some('A') {
                    result.push('X');
                    i += 2;
                } else if next == Some('H') {
                    result.push('X');
                    i += 1;
                } else if next == Some('I') || next == Some('E') || next == Some('Y') {
                    result.push('S');
                } else {
                    result.push('K');
                }
            }
            'D' => {
                if next == Some('G') && matches!(next2, Some('E') | Some('I') | Some('Y')) {
                    result.push('J');
                    i += 2;
                } else {
                    result.push('T');
                }
            }
            'F' => result.push('F'),
            'G' => {
                if next == Some('H') {
                    if i + 2 < len && !is_vowel(chars[i + 2]) {
                        i += 1;
                    } else if i == 0 {
                        result.push('K');
                        i += 1;
                    }
                } else if next == Some('N') {
                    if i + 2 >= len || (i + 2 < len && chars[i + 2] != 'E' && chars[i + 2] != 'D') {
                        // Silent G
                    } else {
                        result.push('K');
                    }
                } else if next == Some('I') || next == Some('E') || next == Some('Y') {
                    result.push('J');
                } else {
                    result.push('K');
                }
            }
            'H' => {
                if i == 0 || !is_vowel(prev.unwrap_or(' ')) {
                    if next.map_or(false, |ch| is_vowel(ch)) {
                        result.push('H');
                    }
                }
            }
            'J' => result.push('J'),
            'K' => {
                if prev != Some('C') {
                    result.push('K');
                }
            }
            'L' => result.push('L'),
            'M' => result.push('M'),
            'N' => result.push('N'),
            'P' => {
                if next == Some('H') {
                    result.push('F');
                    i += 1;
                } else {
                    result.push('P');
                }
            }
            'Q' => result.push('K'),
            'R' => result.push('R'),
            'S' => {
                if next == Some('H') {
                    result.push('X');
                    i += 1;
                } else if next == Some('I') && (next2 == Some('O') || next2 == Some('A')) {
                    result.push('X');
                    i += 2;
                } else {
                    result.push('S');
                }
            }
            'T' => {
                if next == Some('I') && (next2 == Some('O') || next2 == Some('A')) {
                    result.push('X');
                    i += 2;
                } else if next == Some('H') {
                    result.push('0'); // theta
                    i += 1;
                } else if next != Some('C') || next2 != Some('H') {
                    result.push('T');
                }
            }
            'V' => result.push('F'),
            'W' | 'Y' => {
                if next.map_or(false, |ch| is_vowel(ch)) {
                    result.push(c);
                }
            }
            'X' => {
                result.push('K');
                if result.len() < max_length {
                    result.push('S');
                }
            }
            'Z' => result.push('S'),
            _ => {}
        }
        
        i += 1;
    }
    
    result
}

/// Checks if a character is an ASCII vowel (A, E, I, O, U).
///
/// Used by the Metaphone algorithm for phonetic encoding rules.
fn is_vowel(c: char) -> bool {
    matches!(c, 'A' | 'E' | 'I' | 'O' | 'U')
}

/// Check if two strings have the same Metaphone code.
pub fn metaphone_match(a: &str, b: &str) -> bool {
    metaphone(a, 4) == metaphone(b, 4)
}

/// Metaphone similarity based on code matching.
pub fn metaphone_similarity(a: &str, b: &str, max_length: usize) -> f64 {
    let code_a = metaphone(a, max_length);
    let code_b = metaphone(b, max_length);
    
    if code_a.is_empty() || code_b.is_empty() {
        return 0.0;
    }
    
    if code_a == code_b {
        return 1.0;
    }
    
    // Use Jaro-Winkler on the codes for partial matching
    super::jaro::jaro_winkler_similarity(&code_a, &code_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_soundex() {
        assert_eq!(soundex("Robert"), "R163");
        assert_eq!(soundex("Rupert"), "R163");
        assert_eq!(soundex("Rubin"), "R150");
        assert_eq!(soundex("Ashcraft"), "A261");
        assert_eq!(soundex("Ashcroft"), "A261");
    }
    
    #[test]
    fn test_soundex_match() {
        assert!(soundex_match("Robert", "Rupert"));
        assert!(soundex_match("Smith", "Smyth"));
        assert!(!soundex_match("Robert", "Rubin"));
    }
    
    #[test]
    fn test_metaphone() {
        assert_eq!(metaphone("phone", 4), "FN");
        assert_eq!(metaphone("knight", 4), "NT");
    }
    
    #[test]
    fn test_metaphone_match() {
        assert!(metaphone_match("Stephen", "Steven"));
        assert!(metaphone_match("phone", "fone"));
    }
}
