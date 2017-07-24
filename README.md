# wind tunnel data analysis

code for analyzing mosquito and drosophila data collected in a wind tunnel

### Data

Install MySQL database (e.g., through https://www.mamp.info/en/)
(Note: to access phpMyAdmin (database GUI) through the MAMP start page, on MAMP window click Preferences, and change PHP version to 7.0.x)

Make new empty database called “wind_tunnel_db”.

Import database file using:
/path/to/mysql -uusername -ppassword wind_tunnel_db < /path/to/wind_tunnel_db.sql

E.g., on a mac with MAMP using root user:

/Applications/MAMP/Library/bin/mysql -uroot -proot wind_tunnel_db < /Users/me/Downloads/wind_tunnel_db.sql

It’s a large database so it will take some time. If using phpMyAdmin, you can view the database and watch the tables being populated if you want.

Next you will need to create an environment variable containing the connection URI. This should be called WIND_TUNNEL_TEST_DB_CXN_URL and should look like “mysql+mysqldb://username:password@127.0.0.1:8889/wind_tunnel_db”

Do the same for infotaxis db, calling the empty database “infotaxis_db” and populating with the file infotaxis_db.sql . Its environment variable should be called INFOTAXIS_DB_CXN_URL and should look like "mysql+mysqldb://username:password@127.0.0.1:8889/infotaxis_db"

### Code

Getting code to run using Anaconda and Python 2.7:

Download Anaconda.

Create new environment using Python 2.7 and Anaconda:
conda create --name wind_tunnel_env python=2.7 anaconda

Activate new environment:
source activate wind_tunnel_env

Note: if using MAMP, need to add MAMB binaries to path. In .bash_profile in home directory, add line:
export PATH=”/Applications/MAMP/Library/bin:$PATH”
so that when mysql-python connector is installed it knows where to look for mysql_config

Then when env is active install mysql-python (MySQLdb):

pip install mysql-python

Navigate to repository where code lives (or a parent directory).

Run:

jupyter notebook

To open the notebook files. All figures in the manuscript are contained in _paper_draft.ipynb. Plots involving infotaxis simulations require you to first run and save the infotaxis results, which can be done using the code cells in _paper_auxiliary_code.ipynb.

