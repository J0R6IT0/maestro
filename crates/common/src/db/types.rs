use diesel::deserialize::{self, FromSql, FromSqlRow};
use diesel::expression::AsExpression;
use diesel::serialize::{self, IsNull, Output, ToSql};
use diesel::sql_types::Binary;
use diesel::sqlite::{Sqlite, SqliteValue};

#[derive(Debug, Clone, Default, PartialEq, AsExpression, FromSqlRow)]
#[diesel(sql_type = Binary)]
pub struct Embedding(pub Vec<f32>);

#[derive(Debug, Clone, Default, PartialEq, AsExpression, FromSqlRow)]
#[diesel(sql_type = Binary)]
pub struct Neighbors(pub Vec<Neighbor>);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub idx: i32,
    pub distance: f32,
}

impl ToSql<Binary, Sqlite> for Embedding {
    fn to_sql<'b>(&'b self, out: &mut Output<'b, '_, Sqlite>) -> serialize::Result {
        let blob: Vec<u8> = self.0.iter().flat_map(|v| v.to_le_bytes()).collect();
        out.set_value(blob);
        Ok(IsNull::No)
    }
}

impl FromSql<Binary, Sqlite> for Embedding {
    fn from_sql(value: SqliteValue<'_, '_, '_>) -> deserialize::Result<Self> {
        decode_records(value, "embedding", f32::from_le_bytes).map(Self)
    }
}

impl ToSql<Binary, Sqlite> for Neighbors {
    fn to_sql<'b>(&'b self, out: &mut Output<'b, '_, Sqlite>) -> serialize::Result {
        let blob: Vec<u8> = self
            .0
            .iter()
            .flat_map(|n| [n.idx.to_le_bytes(), n.distance.to_le_bytes()])
            .flatten()
            .collect();
        out.set_value(blob);
        Ok(IsNull::No)
    }
}

impl FromSql<Binary, Sqlite> for Neighbors {
    fn from_sql(value: SqliteValue<'_, '_, '_>) -> deserialize::Result<Self> {
        decode_records(value, "neighbors", |b: [u8; 8]| Neighbor {
            idx: i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            distance: f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        })
        .map(Self)
    }
}

/// Splits a blob into `N`-byte records and decodes each one.
fn decode_records<const N: usize, T>(
    value: SqliteValue<'_, '_, '_>,
    kind: &str,
    decode: impl Fn([u8; N]) -> T,
) -> deserialize::Result<Vec<T>> {
    let blob = <Vec<u8> as FromSql<Binary, Sqlite>>::from_sql(value)?;
    let (records, rest) = blob.as_chunks::<N>();
    if !rest.is_empty() {
        return Err(format!("{kind} blob of {} bytes, not a multiple of {N}", blob.len()).into());
    }
    Ok(records.iter().copied().map(decode).collect())
}
