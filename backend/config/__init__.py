import pymysql

# Lets Django's `django.db.backends.mysql` engine work with PyMySQL instead of
# mysqlclient (which needs a C compiler / MySQL dev headers, often painful on
# Windows).
pymysql.install_as_MySQLdb()

# PyMySQL hardcodes version_info=(1, 4, 6, ...) to mimic an old mysqlclient
# release for MySQLdb API compatibility. Django 6 rejects anything below
# 2.2.1, so report a version that satisfies the check.
pymysql.version_info = (2, 2, 4, "final", 0)
pymysql.__version__ = "2.2.4"
