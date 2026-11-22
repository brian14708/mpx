//! Small helpers for editing open-ended JSON wire objects.

use serde_json::{Map, Value};

/// Removes `key`, stringifying non-string values.
pub(crate) fn take_string(map: &mut Map<String, Value>, key: &str) -> Option<String> {
    map.remove(key).map(|value| match value {
        Value::String(value) => value,
        other => other.to_string(),
    })
}

pub(crate) fn take_bool(map: &mut Map<String, Value>, key: &str) -> Option<bool> {
    map.remove(key).and_then(|value| value.as_bool())
}

/// Removes `key` if it holds an object; otherwise returns an empty map.
pub(crate) fn take_object(map: &mut Map<String, Value>, key: &str) -> Map<String, Value> {
    match map.remove(key) {
        Some(Value::Object(fields)) => fields,
        _ => Map::new(),
    }
}

pub(crate) fn insert_option(map: &mut Map<String, Value>, key: &str, value: Option<Value>) {
    if let Some(value) = value {
        map.insert(key.to_owned(), value);
    }
}

/// Inserts `source` entries whose keys are not already present in `target`.
pub(crate) fn merge_missing(target: &mut Map<String, Value>, source: Map<String, Value>) {
    for (key, value) in source {
        target.entry(key).or_insert(value);
    }
}

pub(crate) fn nonempty(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_owned())
}
