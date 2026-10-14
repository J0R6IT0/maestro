use std::{env, fmt::Display, str::FromStr};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("missing env var {0}")]
    Missing(&'static str),
    #[error("invalid env var {name}={value:?}: {reason}")]
    Invalid {
        name: &'static str,
        value: String,
        reason: String,
    },
}

pub fn var(name: &'static str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

pub fn required(name: &'static str) -> Result<String, Error> {
    var(name).ok_or(Error::Missing(name))
}

pub fn var_or(name: &'static str, default: &str) -> String {
    var(name).unwrap_or_else(|| default.into())
}

pub fn parsed<T>(name: &'static str) -> Result<Option<T>, Error>
where
    T: FromStr,
    T::Err: Display,
{
    var(name)
        .map(|value| match value.trim().parse() {
            Ok(parsed) => Ok(parsed),
            Err(e) => Err(invalid(name, value, e)),
        })
        .transpose()
}

pub fn parsed_or<T>(name: &'static str, default: T) -> Result<T, Error>
where
    T: FromStr,
    T::Err: Display,
{
    Ok(parsed(name)?.unwrap_or(default))
}

pub fn list(name: &'static str) -> Vec<String> {
    var(name)
        .iter()
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(String::from)
        .collect()
}

pub fn invalid(name: &'static str, value: impl Into<String>, reason: impl Display) -> Error {
    Error::Invalid {
        name,
        value: value.into(),
        reason: reason.to_string(),
    }
}
