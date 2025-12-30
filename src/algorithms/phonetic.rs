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
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Encode a string to its Soundex code
    #[must_use]
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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_max_length(max_length: usize) -> Self {
        Self { max_length }
    }

    /// Encode a string to its Metaphone code
    #[must_use]
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
///
/// Returns a 4-character code: first letter + 3 digits.
///
/// # Empty String Handling
///
/// Returns an empty string for empty input or input containing no alphabetic
/// characters. This differs from some implementations that return "0000" for
/// empty input, but is consistent with metaphone and other functions in this
/// module.
///
/// # Examples
/// ```
/// use fuzzyrust::algorithms::phonetic::soundex;
/// assert_eq!(soundex("Robert"), "R163");
/// assert_eq!(soundex("Rupert"), "R163");
/// assert_eq!(soundex(""), "");  // Empty input → empty output
/// ```
#[must_use]
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
#[must_use]
pub fn soundex_match(a: &str, b: &str) -> bool {
    soundex(a) == soundex(b)
}

/// Soundex similarity: 1.0 if codes match, 0.0 otherwise.
/// For partial matching, compare individual code positions.
#[must_use]
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
#[must_use]
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
                    if i + 2 >= len || (chars[i + 2] != 'E' && chars[i + 2] != 'D') {
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
                if (i == 0 || !is_vowel(prev.unwrap_or(' '))) && next.is_some_and(is_vowel) {
                    result.push('H');
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
                if next.is_some_and(is_vowel) {
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
#[must_use]
pub fn metaphone_match(a: &str, b: &str) -> bool {
    metaphone(a, 4) == metaphone(b, 4)
}

/// Metaphone similarity based on code matching.
#[must_use]
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

/// Double Metaphone phonetic encoder
///
/// An improvement over Metaphone that returns two codes: primary and alternate.
/// The primary code represents the most likely pronunciation, while the alternate
/// handles variations common in European names (Germanic, Slavic, Italian, etc.).
///
/// # Parameters
/// - `max_length`: Maximum code length for both codes (default: 4)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DoubleMetaphone {
    /// Maximum code length
    pub max_length: usize,
}

impl Default for DoubleMetaphone {
    fn default() -> Self {
        Self { max_length: 4 }
    }
}

impl DoubleMetaphone {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_max_length(max_length: usize) -> Self {
        Self { max_length }
    }

    /// Encode a string to its Double Metaphone codes
    #[must_use]
    pub fn encode(&self, s: &str) -> (String, String) {
        double_metaphone(s, self.max_length)
    }
}

impl Similarity for DoubleMetaphone {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        double_metaphone_similarity(a, b, self.max_length)
    }

    fn name(&self) -> &'static str {
        "double_metaphone"
    }
}

/// Encode a string using the Double Metaphone algorithm.
///
/// Returns a tuple of (primary, alternate) codes. The alternate code may be
/// empty or the same as primary if there's no alternate pronunciation.
///
/// # Examples
/// ```
/// use fuzzyrust::algorithms::phonetic::double_metaphone;
///
/// let (primary, alternate) = double_metaphone("Schmidt", 4);
/// assert_eq!(primary, "XMT");   // German pronunciation
/// assert_eq!(alternate, "SMT"); // Anglicized pronunciation
///
/// let (primary, alternate) = double_metaphone("Smith", 4);
/// assert_eq!(primary, "SM0");   // 0 = theta sound (TH)
/// assert_eq!(alternate, "XMT"); // Alternate
/// ```
#[must_use]
pub fn double_metaphone(s: &str, max_length: usize) -> (String, String) {
    let s = s.trim().to_uppercase();
    let chars: Vec<char> = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();

    if chars.is_empty() {
        return (String::new(), String::new());
    }

    let len = chars.len();
    let mut primary = String::with_capacity(max_length);
    let mut alternate = String::with_capacity(max_length);
    let mut i = 0;

    // Helper closures
    let is_slavo_germanic = |chars: &[char]| -> bool {
        chars.iter().any(|&c| c == 'W' || c == 'K')
            || chars.windows(2).any(|w| w == ['C', 'Z'] || w == ['W', 'I'])
    };

    let slavo_germanic = is_slavo_germanic(&chars);

    // Helper to check for vowels
    let is_vowel_char = |c: char| -> bool { matches!(c, 'A' | 'E' | 'I' | 'O' | 'U') };

    // Helper to get character at position safely
    let char_at = |pos: usize| -> Option<char> { chars.get(pos).copied() };

    // Helper to check if substring matches at position
    let string_at = |pos: usize, patterns: &[&str]| -> bool {
        patterns.iter().any(|&p| {
            let p_chars: Vec<char> = p.chars().collect();
            p_chars
                .iter()
                .enumerate()
                .all(|(j, &pc)| char_at(pos + j) == Some(pc))
        })
    };

    // Handle initial combinations
    if string_at(0, &["GN", "KN", "PN", "WR", "PS"]) {
        i = 1; // Skip first letter
    }

    // Initial 'X' becomes 'S'
    if char_at(0) == Some('X') {
        primary.push('S');
        alternate.push('S');
        i = 1;
    }

    // Main encoding loop
    while i < len && (primary.len() < max_length || alternate.len() < max_length) {
        let c = chars[i];
        let next = char_at(i + 1);
        let prev = if i > 0 { char_at(i - 1) } else { None };
        let next2 = char_at(i + 2);

        // Helper to add codes (respecting max_length)
        let mut add = |p: &str, a: &str| {
            for ch in p.chars() {
                if primary.len() < max_length {
                    primary.push(ch);
                }
            }
            for ch in a.chars() {
                if alternate.len() < max_length {
                    alternate.push(ch);
                }
            }
        };

        match c {
            'A' | 'E' | 'I' | 'O' | 'U' => {
                if i == 0 {
                    add("A", "A");
                }
            }
            'B' => {
                add("P", "P");
                if next == Some('B') {
                    i += 1;
                }
            }
            'C' => {
                // Various C rules
                if string_at(i, &["CIA"]) {
                    add("X", "X");
                    i += 2;
                } else if i == 0 && string_at(i, &["CAESAR"]) {
                    add("S", "S");
                    i += 1;
                } else if string_at(i, &["CH"]) {
                    // Handle CH
                    if i > 0 && string_at(i, &["CHAE"]) {
                        add("K", "X");
                    } else if i == 0
                        && (string_at(i + 1, &["HARAC", "HARIS"])
                            || string_at(i + 1, &["HOR", "HYM", "HIA", "HEM"]))
                    {
                        add("K", "K");
                    } else if string_at(0, &["VAN ", "VON "])
                        || string_at(0, &["SCH"])
                        || string_at(i - 2, &["ORCHES", "ARCHIT", "ORCHID"])
                        || string_at(i + 2, &["T", "S"])
                    {
                        add("K", "K");
                    } else if i == 0 {
                        add("X", "X");
                    } else if string_at(0, &["MC"]) {
                        add("K", "K");
                    } else {
                        add("X", "K");
                    }
                    i += 1;
                } else if string_at(i, &["CZ"]) && !string_at(i - 2, &["WI"]) {
                    add("S", "X");
                    i += 1;
                } else if string_at(i + 1, &["CIA"]) {
                    add("X", "X");
                    i += 2;
                } else if string_at(i, &["CC"]) && !(i == 1 && char_at(0) == Some('M')) {
                    if string_at(i + 2, &["I", "E", "H"]) && !string_at(i + 2, &["HU"]) {
                        if (i == 1 && char_at(0) == Some('A'))
                            || string_at(i - 1, &["UCCEE", "UCCES"])
                        {
                            add("KS", "KS");
                        } else {
                            add("X", "X");
                        }
                        i += 2;
                    } else {
                        add("K", "K");
                        i += 1;
                    }
                } else if string_at(i, &["CK", "CG", "CQ"]) {
                    add("K", "K");
                    i += 1;
                } else if string_at(i, &["CI", "CE", "CY"]) {
                    if string_at(i, &["CIO", "CIE", "CIA"]) {
                        add("S", "X");
                    } else {
                        add("S", "S");
                    }
                    i += 1;
                } else {
                    add("K", "K");
                    if string_at(i + 1, &["C", "K", "G", "Q"]) {
                        i += 1;
                    }
                }
            }
            'D' => {
                if string_at(i, &["DG"]) {
                    if string_at(i + 2, &["I", "E", "Y"]) {
                        add("J", "J");
                        i += 2;
                    } else {
                        add("TK", "TK");
                        i += 1;
                    }
                } else if string_at(i, &["DT", "DD"]) {
                    add("T", "T");
                    i += 1;
                } else {
                    add("T", "T");
                }
            }
            'F' => {
                add("F", "F");
                if next == Some('F') {
                    i += 1;
                }
            }
            'G' => {
                if next == Some('H') {
                    if i > 0 && !is_vowel_char(prev.unwrap_or(' ')) {
                        add("K", "K");
                    } else if i == 0 {
                        if char_at(i + 2) == Some('I') {
                            add("J", "J");
                        } else {
                            add("K", "K");
                        }
                    } else if (i > 1 && string_at(i - 2, &["B", "H", "D"]))
                        || (i > 2 && string_at(i - 3, &["B", "H", "D"]))
                        || (i > 3 && string_at(i - 4, &["B", "H"]))
                    {
                        // Silent GH
                    } else {
                        if i > 2 && prev == Some('U') && string_at(i - 3, &["C", "G", "L", "R", "T"])
                        {
                            add("F", "F");
                        } else if i > 0 && prev != Some('I') {
                            add("K", "K");
                        }
                    }
                    i += 1;
                } else if next == Some('N') {
                    if i == 1 && is_vowel_char(chars[0]) && !slavo_germanic {
                        add("KN", "N");
                    } else if !string_at(i + 2, &["EY"]) && next != Some('Y') && !slavo_germanic {
                        add("N", "KN");
                    } else {
                        add("KN", "KN");
                    }
                    i += 1;
                } else if string_at(i + 1, &["LI"]) && !slavo_germanic {
                    add("KL", "L");
                    i += 1;
                } else if i == 0
                    && (next == Some('Y')
                        || string_at(i + 1, &["ES", "EP", "EB", "EL", "EY", "IB", "IL", "IN", "IE", "EI", "ER"]))
                {
                    add("K", "J");
                } else if (string_at(i + 1, &["ER"]) || next == Some('Y'))
                    && !string_at(0, &["DANGER", "RANGER", "MANGER"])
                    && !string_at(i - 1, &["E", "I"])
                    && !string_at(i - 1, &["RGY", "OGY"])
                {
                    add("K", "J");
                } else if string_at(i + 1, &["E", "I", "Y"]) || string_at(i - 1, &["AGGI", "OGGI"])
                {
                    if string_at(0, &["VAN ", "VON "]) || string_at(0, &["SCH"]) || string_at(i + 1, &["ET"]) {
                        add("K", "K");
                    } else if string_at(i + 1, &["IER"]) {
                        add("J", "J");
                    } else {
                        add("J", "K");
                    }
                } else {
                    add("K", "K");
                    if next == Some('G') {
                        i += 1;
                    }
                }
            }
            'H' => {
                // Only keep if at start or preceded by a consonant and followed by a vowel
                if (i == 0 || !is_vowel_char(prev.unwrap_or(' ')))
                    && next.is_some_and(is_vowel_char)
                {
                    add("H", "H");
                }
            }
            'J' => {
                // Check for Spanish J (pronounced as H/Y)
                if string_at(i, &["JOSE"]) || string_at(0, &["SAN "]) {
                    if (i == 0 && char_at(i + 4) == Some(' ')) || string_at(0, &["SAN "]) {
                        add("H", "H");
                    } else {
                        add("J", "H");
                    }
                } else if i == 0 && !string_at(i, &["JOSE"]) {
                    add("J", "A");
                } else if is_vowel_char(prev.unwrap_or(' '))
                    && !slavo_germanic
                    && (next == Some('A') || next == Some('O'))
                {
                    add("J", "H");
                } else if i == len - 1 {
                    add("J", "");
                } else if !string_at(i + 1, &["L", "T", "K", "S", "N", "M", "B", "Z"])
                    && !string_at(i - 1, &["S", "K", "L"])
                {
                    add("J", "J");
                }
                if next == Some('J') {
                    i += 1;
                }
            }
            'K' => {
                add("K", "K");
                if next == Some('K') {
                    i += 1;
                }
            }
            'L' => {
                if next == Some('L') {
                    // Spanish LL
                    if (i == len - 3 && string_at(i - 1, &["ILLO", "ILLA", "ALLE"]))
                        || ((string_at(len - 2, &["AS", "OS"]) || string_at(len - 1, &["A", "O"]))
                            && string_at(i - 1, &["ALLE"]))
                    {
                        add("L", "");
                    } else {
                        add("L", "L");
                    }
                    i += 1;
                } else {
                    add("L", "L");
                }
            }
            'M' => {
                add("M", "M");
                if (string_at(i - 1, &["UMB"]) && (i + 1 == len - 1 || string_at(i + 2, &["ER"])))
                    || next == Some('M')
                {
                    i += 1;
                }
            }
            'N' => {
                add("N", "N");
                if next == Some('N') {
                    i += 1;
                }
            }
            'P' => {
                if next == Some('H') {
                    add("F", "F");
                    i += 1;
                } else {
                    add("P", "P");
                    if string_at(i + 1, &["P", "B"]) {
                        i += 1;
                    }
                }
            }
            'Q' => {
                add("K", "K");
                if next == Some('Q') {
                    i += 1;
                }
            }
            'R' => {
                // French endings
                if i == len - 1
                    && !slavo_germanic
                    && string_at(i - 2, &["IE"])
                    && !string_at(i - 4, &["ME", "MA"])
                {
                    add("", "R");
                } else {
                    add("R", "R");
                }
                if next == Some('R') {
                    i += 1;
                }
            }
            'S' => {
                if string_at(i - 1, &["ISL", "YSL"]) {
                    // Silent S (island, carlisle)
                } else if i == 0 && string_at(i, &["SUGAR"]) {
                    add("X", "S");
                } else if string_at(i, &["SH"]) {
                    if string_at(i + 1, &["HEIM", "HOEK", "HOLM", "HOLZ"]) {
                        add("S", "S");
                    } else {
                        add("X", "X");
                    }
                    i += 1;
                } else if string_at(i, &["SIO", "SIA"]) || string_at(i, &["SIAN"]) {
                    if !slavo_germanic {
                        add("S", "X");
                    } else {
                        add("S", "S");
                    }
                    i += 2;
                } else if (i == 0 && string_at(i + 1, &["M", "N", "L", "W"]))
                    || string_at(i + 1, &["Z"])
                {
                    add("S", "X");
                    if string_at(i + 1, &["Z"]) {
                        i += 1;
                    }
                } else if string_at(i, &["SC"]) {
                    if char_at(i + 2) == Some('H') {
                        // German SCH
                        if string_at(i + 3, &["OO", "ER", "EN", "UY", "ED", "EM"]) {
                            if string_at(i + 3, &["ER", "EN"]) {
                                add("X", "SK");
                            } else {
                                add("SK", "SK");
                            }
                        } else if i == 0 && !is_vowel_char(char_at(3).unwrap_or(' ')) && char_at(3) != Some('W') {
                            add("X", "S");
                        } else {
                            add("X", "X");
                        }
                        i += 2;
                    } else if string_at(i + 2, &["I", "E", "Y"]) {
                        add("S", "S");
                        i += 2;
                    } else {
                        add("SK", "SK");
                        i += 2;
                    }
                } else {
                    // French final S
                    if i == len - 1 && string_at(i - 2, &["AI", "OI"]) {
                        add("", "S");
                    } else {
                        add("S", "S");
                    }
                    if string_at(i + 1, &["S", "Z"]) {
                        i += 1;
                    }
                }
            }
            'T' => {
                if string_at(i, &["TION"]) || string_at(i, &["TIA", "TCH"]) {
                    add("X", "X");
                    i += 2;
                } else if string_at(i, &["TH"]) || string_at(i, &["TTH"]) {
                    if string_at(i + 2, &["OM", "AM"]) || string_at(0, &["VAN ", "VON "]) || string_at(0, &["SCH"])
                    {
                        add("T", "T");
                    } else {
                        add("0", "T"); // 0 = theta
                    }
                    i += 1;
                } else {
                    add("T", "T");
                    if string_at(i + 1, &["T", "D"]) {
                        i += 1;
                    }
                }
            }
            'V' => {
                add("F", "F");
                if next == Some('V') {
                    i += 1;
                }
            }
            'W' => {
                if string_at(i, &["WR"]) {
                    add("R", "R");
                    i += 1;
                } else if i == 0 && (next.is_some_and(is_vowel_char) || string_at(i, &["WH"])) {
                    if next.is_some_and(is_vowel_char) {
                        add("A", "F");
                    } else {
                        add("A", "A");
                    }
                } else if (i == len - 1 && prev.is_some_and(is_vowel_char))
                    || string_at(i - 1, &["EWSKI", "EWSKY", "OWSKI", "OWSKY"])
                    || string_at(0, &["SCH"])
                {
                    add("", "F");
                } else if string_at(i, &["WICZ", "WITZ"]) {
                    add("TS", "FX");
                    i += 3;
                }
            }
            'X' => {
                if !(i == len - 1
                    && (string_at(i - 3, &["IAU", "EAU"])
                        || string_at(i - 2, &["AU", "OU"])))
                {
                    add("KS", "KS");
                }
                if string_at(i + 1, &["C", "X"]) {
                    i += 1;
                }
            }
            'Z' => {
                if next == Some('H') {
                    // Chinese ZH
                    add("J", "J");
                    i += 1;
                } else if string_at(i + 1, &["ZO", "ZI", "ZA"])
                    || (slavo_germanic && i > 0 && prev != Some('T'))
                {
                    add("S", "TS");
                } else {
                    add("S", "S");
                }
                if next == Some('Z') {
                    i += 1;
                }
            }
            _ => {}
        }

        i += 1;
    }

    // If alternate is identical to primary, clear it
    if alternate == primary {
        alternate.clear();
    }

    (primary, alternate)
}

/// Check if two strings match using Double Metaphone.
///
/// Returns true if any of the codes match (primary-primary, primary-alternate,
/// alternate-primary, or alternate-alternate).
#[must_use]
pub fn double_metaphone_match(a: &str, b: &str) -> bool {
    let (a_primary, a_alternate) = double_metaphone(a, 4);
    let (b_primary, b_alternate) = double_metaphone(b, 4);

    if a_primary.is_empty() || b_primary.is_empty() {
        return false;
    }

    // Check all combinations
    if a_primary == b_primary {
        return true;
    }
    if !a_alternate.is_empty() && a_alternate == b_primary {
        return true;
    }
    if !b_alternate.is_empty() && a_primary == b_alternate {
        return true;
    }
    if !a_alternate.is_empty() && !b_alternate.is_empty() && a_alternate == b_alternate {
        return true;
    }

    false
}

/// Double Metaphone similarity based on code matching.
///
/// Returns 1.0 for exact match, 0.0 for no match, or a partial score
/// based on Jaro-Winkler similarity of the best matching codes.
#[must_use]
pub fn double_metaphone_similarity(a: &str, b: &str, max_length: usize) -> f64 {
    let (a_primary, a_alternate) = double_metaphone(a, max_length);
    let (b_primary, b_alternate) = double_metaphone(b, max_length);

    if a_primary.is_empty() || b_primary.is_empty() {
        return 0.0;
    }

    // Collect all non-empty codes
    let a_codes: Vec<&str> = [a_primary.as_str(), a_alternate.as_str()]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();
    let b_codes: Vec<&str> = [b_primary.as_str(), b_alternate.as_str()]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();

    // Find best match among all code combinations
    let mut best_score = 0.0;
    for a_code in &a_codes {
        for b_code in &b_codes {
            if a_code == b_code {
                return 1.0; // Exact match
            }
            let score = super::jaro::jaro_winkler_similarity(a_code, b_code);
            if score > best_score {
                best_score = score;
            }
        }
    }

    best_score
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

    #[test]
    fn test_double_metaphone() {
        // Test Germanic names with alternate encodings
        let (primary, alternate) = double_metaphone("Schmidt", 4);
        assert!(!primary.is_empty());
        // Schmidt should have different primary and alternate

        // Test basic encoding
        let (primary, _) = double_metaphone("Smith", 4);
        assert!(!primary.is_empty());

        // Test empty string
        let (primary, alternate) = double_metaphone("", 4);
        assert!(primary.is_empty());
        assert!(alternate.is_empty());

        // Test single character
        let (primary, _) = double_metaphone("A", 4);
        assert_eq!(primary, "A");
    }

    #[test]
    fn test_double_metaphone_match() {
        // Names that sound similar
        assert!(double_metaphone_match("Stephen", "Steven"));
        assert!(double_metaphone_match("Katherine", "Catherine"));

        // Names that don't match
        assert!(!double_metaphone_match("John", "Mary"));
    }

    #[test]
    fn test_double_metaphone_similarity() {
        // Exact phonetic match
        let sim = double_metaphone_similarity("Stephen", "Steven", 4);
        assert!(sim > 0.8);

        // Different names
        let sim = double_metaphone_similarity("John", "Mary", 4);
        assert!(sim < 0.5);

        // Empty strings
        let sim = double_metaphone_similarity("", "test", 4);
        assert_eq!(sim, 0.0);
    }
}
