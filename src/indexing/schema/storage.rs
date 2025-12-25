///! Column-oriented storage for multi-field records
///!
///! This module provides optimized storage structures for multi-field records.
///! Key features:
///! - Column-oriented layout for better cache locality
///! - Efficient field access without full record deserialization
///! - Support for different field types (text, tokens)

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during column operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ColumnError {
    /// Attempted to push wrong type to column
    #[error("Type mismatch: attempted to push {attempted} to {expected} column")]
    TypeMismatch {
        expected: String,
        attempted: String,
    },

    /// Attempted to get wrong type from column
    #[error("Type mismatch: attempted to get {attempted} from {expected} column")]
    GetTypeMismatch {
        expected: String,
        attempted: String,
    },

    /// Index out of bounds
    #[error("Index {0} out of bounds for column of length {1}")]
    IndexOutOfBounds(usize, usize),
}

/// A record in a multi-field index
///
/// Represents a single document/entity with multiple named fields.
/// Fields are stored as key-value pairs where values are strings.
///
/// # Example
///
/// ```rust
/// use fuzzyrust::indexing::schema::storage::Record;
///
/// let mut record = Record::new();
/// record.set_field("name", "MacBook Pro");
/// record.set_field("tags", "laptop,apple,computing");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Field name -> value mapping
    pub fields: AHashMap<String, String>,

    /// Optional user data (e.g., database ID)
    pub data: Option<u64>,
}

impl Record {
    /// Create a new empty record
    pub fn new() -> Self {
        Self {
            fields: AHashMap::new(),
            data: None,
        }
    }

    /// Create a new record with user data
    pub fn with_data(data: u64) -> Self {
        Self {
            fields: AHashMap::new(),
            data: Some(data),
        }
    }

    /// Set a field value
    pub fn set_field(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.fields.insert(name.into(), value.into());
    }

    /// Get a field value
    pub fn get_field(&self, name: &str) -> Option<&String> {
        self.fields.get(name)
    }

    /// Check if a field exists
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// Remove a field
    pub fn remove_field(&mut self, name: &str) -> Option<String> {
        self.fields.remove(name)
    }

    /// Get number of fields
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Get all field names
    pub fn field_names(&self) -> Vec<&String> {
        self.fields.keys().collect()
    }
}

impl Default for Record {
    fn default() -> Self {
        Self::new()
    }
}

/// Column storage for a single field across all records
///
/// Stores values for one field in a contiguous vector for cache efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldColumn {
    /// Text field: stores strings directly
    Text(Vec<Option<String>>),

    /// Token set field: stores tokenized values
    /// Each record's value is split into tokens (e.g., CSV -> Vec<String>)
    Tokens(Vec<Option<Vec<String>>>),
}

impl FieldColumn {
    /// Create a new empty text column
    pub fn new_text() -> Self {
        FieldColumn::Text(Vec::new())
    }

    /// Create a new empty token column
    pub fn new_tokens() -> Self {
        FieldColumn::Tokens(Vec::new())
    }

    /// Add a text value to the column
    ///
    /// # Errors
    ///
    /// Returns `ColumnError::TypeMismatch` if called on a non-text column.
    pub fn push_text(&mut self, value: Option<String>) -> Result<(), ColumnError> {
        match self {
            FieldColumn::Text(vec) => {
                vec.push(value);
                Ok(())
            }
            FieldColumn::Tokens(_) => Err(ColumnError::TypeMismatch {
                expected: "Tokens".to_string(),
                attempted: "Text".to_string(),
            }),
        }
    }

    /// Add tokenized value to the column
    ///
    /// # Errors
    ///
    /// Returns `ColumnError::TypeMismatch` if called on a non-token column.
    pub fn push_tokens(&mut self, tokens: Option<Vec<String>>) -> Result<(), ColumnError> {
        match self {
            FieldColumn::Tokens(vec) => {
                vec.push(tokens);
                Ok(())
            }
            FieldColumn::Text(_) => Err(ColumnError::TypeMismatch {
                expected: "Text".to_string(),
                attempted: "Tokens".to_string(),
            }),
        }
    }

    /// Get text value at index
    ///
    /// # Errors
    ///
    /// Returns `ColumnError::GetTypeMismatch` if called on a non-text column.
    /// Returns `ColumnError::IndexOutOfBounds` if index is out of range.
    pub fn get_text(&self, index: usize) -> Result<Option<&String>, ColumnError> {
        match self {
            FieldColumn::Text(vec) => {
                if index >= vec.len() {
                    Err(ColumnError::IndexOutOfBounds(index, vec.len()))
                } else {
                    Ok(vec.get(index).and_then(|opt| opt.as_ref()))
                }
            }
            FieldColumn::Tokens(_) => Err(ColumnError::GetTypeMismatch {
                expected: "Tokens".to_string(),
                attempted: "Text".to_string(),
            }),
        }
    }

    /// Get tokens at index
    ///
    /// # Errors
    ///
    /// Returns `ColumnError::GetTypeMismatch` if called on a non-token column.
    /// Returns `ColumnError::IndexOutOfBounds` if index is out of range.
    pub fn get_tokens(&self, index: usize) -> Result<Option<&Vec<String>>, ColumnError> {
        match self {
            FieldColumn::Tokens(vec) => {
                if index >= vec.len() {
                    Err(ColumnError::IndexOutOfBounds(index, vec.len()))
                } else {
                    Ok(vec.get(index).and_then(|opt| opt.as_ref()))
                }
            }
            FieldColumn::Text(_) => Err(ColumnError::GetTypeMismatch {
                expected: "Text".to_string(),
                attempted: "Tokens".to_string(),
            }),
        }
    }

    /// Get the number of entries in this column
    pub fn len(&self) -> usize {
        match self {
            FieldColumn::Text(vec) => vec.len(),
            FieldColumn::Tokens(vec) => vec.len(),
        }
    }

    /// Check if column is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Optimized column-oriented storage for multi-field records
///
/// Instead of storing records as rows (record -> fields), we store them as columns
/// (field -> all values). This provides better cache locality when accessing a single
/// field across many records.
///
/// # Example
///
/// ```rust
/// use fuzzyrust::indexing::schema::storage::OptimizedStorage;
/// use fuzzyrust::indexing::schema::types::FieldType;
///
/// let mut storage = OptimizedStorage::new();
/// storage.add_column("name", FieldType::ShortText {
///     max_length: 100,
///     default_algorithm: Default::default(),
/// });
/// storage.add_column("tags", FieldType::TokenSet {
///     separator: ",".to_string(),
///     default_algorithm: Default::default(),
/// });
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedStorage {
    /// Field name -> column index mapping
    field_indices: AHashMap<String, usize>,

    /// Column data (one per field)
    columns: Vec<FieldColumn>,

    /// Field names (in same order as columns)
    field_names: Vec<String>,

    /// User data for each record (parallel to column length)
    record_data: Vec<Option<u64>>,

    /// Number of records stored
    record_count: usize,
}

impl OptimizedStorage {
    /// Create a new empty storage
    pub fn new() -> Self {
        Self {
            field_indices: AHashMap::new(),
            columns: Vec::new(),
            field_names: Vec::new(),
            record_data: Vec::new(),
            record_count: 0,
        }
    }

    /// Add a column for a field
    pub fn add_column(&mut self, field_name: impl Into<String>, field_type: super::types::FieldType) {
        let field_name = field_name.into();
        let index = self.columns.len();

        // Create appropriate column type based on field type
        let column = match field_type {
            super::types::FieldType::ShortText { .. } => FieldColumn::new_text(),
            super::types::FieldType::LongText { .. } => FieldColumn::new_text(),
            super::types::FieldType::TokenSet { .. } => FieldColumn::new_tokens(),
        };

        self.field_indices.insert(field_name.clone(), index);
        self.field_names.push(field_name);
        self.columns.push(column);
    }

    /// Add a record to storage
    ///
    /// The record's fields are distributed into their respective columns.
    /// Missing fields are stored as None.
    ///
    /// # Errors
    ///
    /// Returns `ColumnError` if there's a type mismatch when adding field values.
    pub fn add_record(&mut self, record: &Record) -> Result<usize, ColumnError> {
        let record_id = self.record_count;

        // Add values to each column
        for (field_name, column) in self.field_names.iter().zip(self.columns.iter_mut()) {
            let value = record.get_field(field_name);

            match column {
                FieldColumn::Text(_) => {
                    column.push_text(value.cloned())?;
                }
                FieldColumn::Tokens(_) => {
                    let tokens = value.map(|v| {
                        // Simple CSV tokenization (could be made configurable)
                        v.split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect()
                    });
                    column.push_tokens(tokens)?;
                }
            }
        }

        // Add user data
        self.record_data.push(record.data);

        self.record_count += 1;
        Ok(record_id)
    }

    /// Get a record by ID
    ///
    /// Reconstructs the record from column storage.
    ///
    /// # Errors
    ///
    /// Returns `ColumnError` if there's a type mismatch or index out of bounds.
    pub fn get_record(&self, id: usize) -> Result<Option<Record>, ColumnError> {
        if id >= self.record_count {
            return Ok(None);
        }

        let mut record = Record::new();
        record.data = self.record_data[id];

        // Collect values from each column
        for (field_name, column) in self.field_names.iter().zip(self.columns.iter()) {
            match column {
                FieldColumn::Text(_) => {
                    if let Some(value) = column.get_text(id)? {
                        record.set_field(field_name.clone(), value.clone());
                    }
                }
                FieldColumn::Tokens(_) => {
                    if let Some(tokens) = column.get_tokens(id)? {
                        // Reconstruct comma-separated string
                        let value = tokens.join(", ");
                        record.set_field(field_name.clone(), value);
                    }
                }
            }
        }

        Ok(Some(record))
    }

    /// Get a field value for a specific record
    ///
    /// # Returns
    ///
    /// - `Ok(Some(value))` if the field exists and has a value
    /// - `Ok(None)` if the record or field is not found (unknown field name, out of bounds)
    /// - Note: This returns Ok(None) for missing data, consistent with Option-style lookups.
    ///   For stricter error handling, use `get_record()` instead.
    pub fn get_field_value(&self, record_id: usize, field_name: &str) -> Option<String> {
        let field_idx = self.field_indices.get(field_name)?;
        // Use bounds-checked access instead of direct indexing
        let column = self.columns.get(*field_idx)?;

        match column {
            FieldColumn::Text(vec) => vec.get(record_id)?.as_ref().cloned(),
            FieldColumn::Tokens(vec) => {
                vec.get(record_id)?
                    .as_ref()
                    .map(|tokens| tokens.join(", "))
            }
        }
    }

    /// Get the number of records
    pub fn len(&self) -> usize {
        self.record_count
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.record_count == 0
    }

    /// Get the number of columns (fields)
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Get all field names
    pub fn field_names(&self) -> &[String] {
        &self.field_names
    }

    /// Get user data for a record
    pub fn get_record_data(&self, id: usize) -> Option<u64> {
        self.record_data.get(id).and_then(|&opt| opt)
    }
}

impl Default for OptimizedStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexing::schema::types::{Algorithm, FieldType};

    #[test]
    fn test_record_basic() {
        let mut record = Record::new();
        record.set_field("name", "John Doe");
        record.set_field("email", "john@example.com");

        assert_eq!(record.field_count(), 2);
        assert_eq!(record.get_field("name").unwrap(), "John Doe");
        assert_eq!(record.get_field("email").unwrap(), "john@example.com");
        assert!(record.get_field("unknown").is_none());
    }

    #[test]
    fn test_record_with_data() {
        let record = Record::with_data(12345);
        assert_eq!(record.data, Some(12345));
    }

    #[test]
    fn test_field_column_text() {
        let mut column = FieldColumn::new_text();
        column.push_text(Some("value1".to_string())).unwrap();
        column.push_text(None).unwrap();
        column.push_text(Some("value3".to_string())).unwrap();

        assert_eq!(column.len(), 3);
        assert_eq!(column.get_text(0).unwrap().unwrap(), "value1");
        assert!(column.get_text(1).unwrap().is_none());
        assert_eq!(column.get_text(2).unwrap().unwrap(), "value3");
    }

    #[test]
    fn test_field_column_tokens() {
        let mut column = FieldColumn::new_tokens();
        column.push_tokens(Some(vec!["tag1".to_string(), "tag2".to_string()])).unwrap();
        column.push_tokens(None).unwrap();

        assert_eq!(column.len(), 2);
        assert_eq!(column.get_tokens(0).unwrap().unwrap().len(), 2);
        assert!(column.get_tokens(1).unwrap().is_none());
    }

    #[test]
    fn test_optimized_storage() {
        let mut storage = OptimizedStorage::new();
        storage.add_column(
            "name",
            FieldType::ShortText {
                max_length: 100,
                default_algorithm: Algorithm::Levenshtein,
            },
        );
        storage.add_column(
            "tags",
            FieldType::TokenSet {
                separator: ",".to_string(),
                default_algorithm: Algorithm::Jaccard,
            },
        );

        // Add records
        let mut record1 = Record::new();
        record1.set_field("name", "Alice");
        record1.set_field("tags", "developer, python");
        let id1 = storage.add_record(&record1).unwrap();

        let mut record2 = Record::new();
        record2.set_field("name", "Bob");
        record2.set_field("tags", "designer, ui");
        let id2 = storage.add_record(&record2).unwrap();

        assert_eq!(storage.len(), 2);
        assert_eq!(storage.column_count(), 2);

        // Retrieve records
        let retrieved1 = storage.get_record(id1).unwrap().unwrap();
        assert_eq!(retrieved1.get_field("name").unwrap(), "Alice");

        let retrieved2 = storage.get_record(id2).unwrap().unwrap();
        assert_eq!(retrieved2.get_field("name").unwrap(), "Bob");

        // Get specific field value
        assert_eq!(
            storage.get_field_value(id1, "name").unwrap(),
            "Alice"
        );
    }

    #[test]
    fn test_storage_with_missing_fields() {
        let mut storage = OptimizedStorage::new();
        storage.add_column(
            "name",
            FieldType::ShortText {
                max_length: 100,
                default_algorithm: Algorithm::Levenshtein,
            },
        );
        storage.add_column(
            "email",
            FieldType::ShortText {
                max_length: 100,
                default_algorithm: Algorithm::Levenshtein,
            },
        );

        // Record with only one field
        let mut record = Record::new();
        record.set_field("name", "Alice");
        // No email provided

        let id = storage.add_record(&record).unwrap();
        let retrieved = storage.get_record(id).unwrap().unwrap();

        assert_eq!(retrieved.get_field("name").unwrap(), "Alice");
        assert!(retrieved.get_field("email").is_none());
    }
}
