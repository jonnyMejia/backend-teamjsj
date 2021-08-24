import psycopg2
from app.core.config import DB_NAME, DB_HOST, DB_PORT, DB_USER, DB_PASSWORD

class Database:

    def __init__(self, dbname=DB_NAME, host=DB_HOST, port=DB_PORT, user=DB_USER, passwd=DB_PASSWORD):
        self.dbname = dbname
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.conn = psycopg2.connect(dbname=self.dbname, host=self.host, port=self.port,
                                   user=self.user, password=self.passwd)
        self.cur =  self.conn.cursor()
        
    def close(self):
        self.cur.close()
        self.conn.close()

def selectQuery(query):

    db = Database()

    try:

        db.cur.execute(query)
        columnsNames = [column[0] for column in db.cur.description]

    except (Exception, psycopg2.Error) as error:
    
        print("Error in selecting the data:", error)
    
    return (db.cur.fetchall(), columnsNames)
