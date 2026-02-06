#!/bin/sh
sqlite3 data/db.sqlite "SELECT level_id, COUNT(*) FROM episodes GROUP BY level_id;"
