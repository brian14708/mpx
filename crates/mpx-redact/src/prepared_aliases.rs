use std::{
    collections::{HashMap, HashSet},
    ptr,
};

use aho_corasick::AhoCorasick;
use memchr::memmem;

use crate::{
    Error,
    replacement_template::{bounded, contains_alias},
    rules::RuleMatch,
};

pub(crate) struct AliasCandidate<'a> {
    pub(crate) text: &'a str,
    pub(crate) collides: bool,
}

pub(crate) struct PreparedAliases {
    candidates: Vec<String>,
    collisions: Vec<bool>,
    matches: std::vec::IntoIter<Option<usize>>,
}

impl PreparedAliases {
    pub(crate) fn new(
        text: &str,
        spans: &[RuleMatch<'_>],
        reserved: &HashSet<String>,
        seed_for: impl Fn(&str) -> [u8; 32],
    ) -> Result<Self, Error> {
        let matches = spans.iter().filter_map(|matched| {
            let (template, reversible) = matched.replacement.as_template()?;
            (reversible && !template.framed()).then(|| (template, &text[matched.span.clone()]))
        });
        let mut output = String::new();
        let mut slot = String::new();
        let mut previous = None;
        let mut ids = HashMap::new();
        let mut by_match = Vec::new();
        for (template, value) in matches {
            let repeated = previous.filter(|&(last_template, last_value, _)| {
                ptr::eq(template, last_template) && value == last_value
            });
            let id = if let Some((_, _, id)) = repeated {
                id
            } else if template.generate(&seed_for(value), &mut output, &mut slot) {
                let next_id = ids.len();
                Some(*ids.entry(output.clone()).or_insert(next_id))
            } else {
                None
            };
            previous = Some((template, value, id));
            by_match.push(id);
        }
        let mut candidates = vec![String::new(); ids.len()];
        for (alias, id) in ids {
            candidates[id] = alias;
        }
        let patterns: Vec<_> = candidates.iter().map(String::as_str).collect();
        let collisions = alias_collisions(&patterns, reserved)?;
        Ok(Self {
            candidates,
            collisions,
            matches: by_match.into_iter(),
        })
    }

    // One call per alias span, in the same order used to prepare candidates.
    pub(crate) fn next(&mut self) -> Option<AliasCandidate<'_>> {
        let id = self.matches.next().flatten()?;
        Some(AliasCandidate {
            text: &self.candidates[id],
            collides: self.collisions[id],
        })
    }
}

fn has_reserved_alias(texts: &HashSet<String>, alias: &str) -> bool {
    texts.iter().any(|text| contains_alias(text, alias))
}

fn alias_collisions(patterns: &[&str], texts: &HashSet<String>) -> Result<Vec<bool>, Error> {
    if let [only] = patterns {
        return Ok(vec![has_reserved_alias(texts, only)]);
    }
    if patterns.is_empty() || texts.is_empty() {
        return Ok(vec![false; patterns.len()]);
    }
    // Skip fields that lack a suffix shared by every candidate. In the common
    // case of a generated domain absent from the input, this avoids a build.
    let suffix = patterns[1..]
        .iter()
        .fold(patterns[0].as_bytes(), |suffix, pattern| {
            let length = suffix
                .iter()
                .rev()
                .zip(pattern.bytes().rev())
                .take_while(|&(left, right)| *left == right)
                .count();
            &suffix[suffix.len() - length..]
        });
    let finder = memmem::Finder::new(suffix);
    let mut texts = texts
        .iter()
        .filter(|text| finder.find(text.as_bytes()).is_some())
        .peekable();
    let mut collisions = vec![false; patterns.len()];
    if texts.peek().is_none() {
        return Ok(collisions);
    }
    let automaton = AhoCorasick::new(patterns).map_err(|error| Error::DetectorFailed {
        message: format!("could not build alias automaton: {error}"),
    })?;
    let mut remaining = patterns.len();
    for text in texts {
        for found in automaton.find_overlapping_iter(text) {
            let collision = &mut collisions[found.pattern().as_usize()];
            if !*collision && bounded(text, found.range()) {
                *collision = true;
                remaining -= 1;
                if remaining == 0 {
                    return Ok(collisions);
                }
            }
        }
    }
    Ok(collisions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_alias_collisions_respect_value_boundaries() {
        let texts = HashSet::from([
            "pré.café a..b...c....d a@b.fr 末+木 .dot.".to_owned(),
            "start escaped% _value end".to_owned(),
        ]);
        let present = ["a", "b", "c", "d", "a@b.fr", "末+木", "dot", "start", "end"];
        let absent = ["pré", "café", "a@b", "fr", "末", "木", "escaped", "value"];
        let patterns = [present.as_slice(), absent.as_slice()].concat();
        let collisions = alias_collisions(&patterns, &texts).unwrap();
        for (&pattern, found) in patterns.iter().zip(collisions) {
            assert_eq!(found, present.contains(&pattern), "{pattern}");
        }
    }
}
