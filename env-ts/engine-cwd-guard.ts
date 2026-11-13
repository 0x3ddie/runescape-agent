/**
 * env-ts/engine-cwd-guard.ts: fail fast when the engine is imported from the
 * wrong working directory.
 *
 * The engine reads its data with paths relative to cwd: `World.ts` opens
 * `data/config/private.pem` while its module body runs, and the cache config
 * modules create `data/pack/` before that. Run an engine-importing script from
 * anywhere else and you get an ENOENT about a pem file, plus a stray `data/` and
 * `db.sqlite` written into whatever directory you were standing in.
 *
 * Import this FIRST, before any `vendor/rs-sdk/server/engine/src/...` import.
 * Module bodies evaluate in source order, so this check runs before the engine
 * can touch the filesystem.
 */

import fs from 'node:fs';

const REQUIRED = 'data/config/private.pem';

if (!fs.existsSync(REQUIRED)) {
    console.error(`\nWrong working directory: ${process.cwd()}`);
    console.error(`Expected to find ${REQUIRED} relative to cwd.`);
    console.error('\nThe engine resolves its data relative to cwd, so run from the engine package:\n');
    console.error('  cd vendor/rs-sdk/server/engine');
    console.error('  bun run ../../../../env-ts/engine-probe.ts\n');
    process.exit(2);
}
