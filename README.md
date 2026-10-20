# maestro

Experimental music sonic similarity engine for Navidrome (and other OpenSubsonic servers), based on [MAEST](https://essentia.upf.edu/models.html#maest) embeddings.

maestro is made to be as simple and lightweight as possible, providing only the basic features the average listener needs. Currently, that means the features the OpenSubsonic's [sonic similarity extension](https://opensubsonic.netlify.app/docs/extensions/sonicsimilarity/) supports: similar tracks and path between two songs.

It can handle very large music libraries efficiently and supports CUDA and OpenVINO GPU acceleration.

There is no UI and no extra features. If you need more advanced features, such as lyrics analysis or AI-based playlist generation, it is recommended to use [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI) instead.

## Setup

### Service

1. Install docker.
2. Grab the latest `docker-compose.yml` file from the [releases page](https://github.com/J0R6IT0/maestro/releases).
3. See the [configuration](#configuration) section for the available environment variables.
4. Start the container. It should perform the initial scan which may take a few days depending on your library size and hardware.

### Navidrome plugin

1. Download the latest `maestro.ndp` file from the [releases page](https://github.com/J0R6IT0/maestro/releases).
2. Place the `maestro.ndp` file in the Navidrome plugins directory.
3. If `Plugins.AutoReload` is disabled, restart Navidrome.
4. Enable the plugin and specify the server address.

The plugin works in two modes, which can be used together:

- **Sonic similarity**: implements the OpenSubsonic [sonic similarity extension](https://opensubsonic.netlify.app/docs/extensions/sonicsimilarity/) (`getSonicSimilarTracks` and `findSonicPath`), used by clients that support it. It doesn't require any configuration.
- **Similar tracks**: for the standard `getSimilarSongs` endpoint, so clients without the extension also get sonically similar tracks. It requires adding `maestro` to the Navidrome [`Agents`](https://www.navidrome.org/docs/usage/configuration-options/) option (`ND_AGENTS` env variable). Agents are tried in order and the first one that returns tracks wins, so put it first to prefer maestro and fall back to the others for tracks it hasn't analyzed yet, e.g. `ND_AGENTS=maestro,deezer,lastfm,listenbrainz`.

## Configuration

The following environment variables can be added to the compose file or to a .env if running outside docker.

| Variable | Default | Values | Description |
|---|---|---|---|
| `TZ` | `UTC` | [time zone](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones), e.g. `Europe/Madrid` | Time zone of `ANALYZE_CRON`. Docker only. |
| `ANALYZE_CRON` | `0 2 * * *` | cron expression | When the analyzer runs, after a first pass at startup. Docker only. |
| `PROVIDER` | `subsonic` | `subsonic` | Your music library provider. |
| `SUBSONIC_URL` | required | URL | Address of the OpenSubsonic server. |
| `SUBSONIC_USER` | required | | User to log in with. |
| `SUBSONIC_PASSWORD` | required | | Password of that user. |
| `SUBSONIC_FORMAT` | `opus` | any format the server transcodes to | Format the server transcodes tracks to before they are analyzed. Only used if the server doesn't support the OpenSubsonic [transcoding](https://opensubsonic.netlify.app/docs/extensions/transcoding/) extension. |
| `DOWNLOAD_WORKERS` | `4` | number | Parallel downloads and decodes. |
| `EXCLUDED_LIBRARIES` | none | comma-separated names or ids | Libraries to not analyze and leave out of results (e.g. `Audiobooks, Podcasts`). |
| `DEDUP_STRATEGY` | `title+artist` | `title+artist`, `title`, `off` | Which tracks count as duplicates by title: same title and a shared artist, same title only, or none. Tracks with the same audio are always duplicates. |
| `DEDUP_LEVEL` | `recording` | `recording`, `song` | How alike duplicates by title must sound: the same recording (remasters, edits, compilation copies), or any version of the song (also remixes, live takes). Note that `title` strategy with `song` level may also merge unrelated songs that share a common title. |
| `MAX_DURATION_INCREASE` | no limit | percentage, e.g. `50%` | Exclude from results tracks longer than the seed by more than this (for `/path`, than the longer of the two). |
| `MIN_SIMILARITY` | `0%` | percentage, e.g. `60%` | Leave out of `/similar` results tracks whose `similarity` is below this. In general it is **NOT** recommended to set this variable unless your library is small or you have a specific use case. |
| `LOG_LEVEL` | `info,ort=warn` | `error`, `warn`, `info`, `debug`, `trace` or a [filter](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html#directives) | Log verbosity. |

## How it works

There are 2 main components: the analyzer and the server.

### Analyzer

1. Each time the analyzer runs (`ANALYZE_CRON`), it syncs the library, updates metadata and analyzes new tracks.
2. Each track is decoded and processed by the MAEST model. The model produces a 2304-d embedding that describes the sound. Tracks that sound alike get embeddings that point in a similar direction, so their distance is small.
3. Every new embedding is compared with all the others and the 100 nearest neighbors are stored.
4. The analyzer also writes a search index (`maestro.search`) next to the database. It stores each track's embedding compressed to 128 numbers (its position along the library's 128 most important directions) and links to the 16 nearest neighbors. Tracks that sound alike are stored close to each other, so on large libraries, getting a path only reads a small part of the file.

### Server

The server exposes two endpoints:

- `/similar` reads the nearest-neighbor index built in [step 3 of the Analyzer](#analyzer) and returns up to 100 similar tracks, leaving out those below `MIN_SIMILARITY`.
- `/path` returns a path up to N songs between two tracks using the search index built in [step 4 of the Analyzer](#analyzer). A path from song A to song B follows the neighbor links. It first finds the chain of links between A and B with the smallest steps, then evenly spaces N points along that chain and picks one track near each point, choosing them together so that every step is small and about the same size.

## License

maestro's source code is licensed under the [MIT License](LICENSE).

The MAEST model ([`discogs-maest-30s-pw-519l-2`](https://essentia.upf.edu/models.html#maest)) is not part of the source code and is not covered by the MIT License. It is created by the [Music Technology Group](https://www.upf.edu/web/mtg) (Universitat Pompeu Fabra) and licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The Docker images include it, unmodified, as downloaded from [Essentia](https://essentia.upf.edu/models.html), so they can't be used for commercial purposes without a [proprietary license](https://www.upf.edu/web/mtg/contact) from the MTG.

```bibtex
@inproceedings{alonso2023,
  title={Efficient Supervised Training of Audio Transformers for Music Representation Learning},
  author={Alonso-Jim{\'e}nez, Pablo and Serra, Xavier and Bogdanov, Dmitry},
  booktitle={Conference of the International Society for Music Information Retrieval (ISMIR)},
  year={2023}
}
```
