"""
Copy everything from the raw database into the analysis database.
"""
from __future__ import print_function, division
import os
from subprocess import call
from db_api.connect import TEST

DB_BACKUP_DIRECTORY = '/Users/rkp/Dropbox/database_backups/wind_tunnel'

if TEST:
    db_name_raw = 'wind_tunnel_raw_sample_db'
    db_name_analysis = 'wind_tunnel_test_db'
else:
    db_name_raw = 'wind_tunnel_raw_db'
    db_name_analysis = 'wind_tunnel_db'

# download raw database
dump_command = ' '.join(['/Applications/MAMP/Library/Bin/mysqldump', '-uwind_tunnel_user',
                         '-pwind_tunnel_pass', db_name_raw, '>',
                         os.path.join(DB_BACKUP_DIRECTORY, db_name_raw + '.sql')])
print("Dumping raw database with command '{}'".format(dump_command))
call(dump_command, shell=True)

# upload downloaded raw database into analysis database
populate_command = ' '.join(['/Applications/MAMP/Library/Bin/mysql', '-uwind_tunnel_user',
                             '-pwind_tunnel_pass', db_name_analysis, '<',
                             os.path.join(DB_BACKUP_DIRECTORY, db_name_raw + '.sql')])
print("Populating analysis database with command '{}'".format(populate_command))
call(populate_command, shell=True)