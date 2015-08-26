__author__ = 'surajman'
# MAKE "CONN" OBJECTS TO CONTROL CONNECTION TO DB. 1 OBJ USES 1 CONN,CUR.
from gensim.models import Word2Vec as W
import sqlite3

class SQLCon:
    def __init__(self, db_path='w2v.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cur  = self.conn.cursor()

    def write(self, model_path, dim):
        model = W.load(model_path)
        words = model.vocab.keys()
        conn = self.conn
        cur = self.cur

        create_q = "CREATE TABLE %s (word text," % 'array'
        for i in range(dim):
            create_q += "D%d real," % (i)
        create_q = create_q[:-1]+")"

        insert_q = "INSERT INTO %s VALUES (" % 'array'
        for i in range(dim+1):insert_q += "?,"
        insert_q = insert_q[:-1]+")"

        cur.execute(create_q)
        conn.commit()
        inp =[]
        for idx,word in enumerate(words):
            inp.append((word,) + tuple(model[word].tolist()))
            if idx % 10000 == 0:
                cur.execute(insert_q,inp)
                conn.commit()
                del inp
                inp=[]
                continue
        cur.execute("CREATE UNIQUE INDEX idx_word ON %s(word)" % 'array')
        cur.close()
        conn.close()

    # Returns word-vector from the Word2Vec SQLite db
    def read(self, word):
        conn = self.conn
        cur = self.cur
        colQuery = "PRAGMA table_info(%s)" % 'array'
        cur.execute(colQuery)
        nCol = len(cur.fetchall())-1
        vecQuery=""
        for i in range (nCol):
            vecQuery += " D%d," % i
        vecQuery = "SELECT" + vecQuery[:-1] + " FROM array WHERE word = '%s'"

        vecQ = vecQuery % word
        cur.execute(vecQ)
        conn.commit()
        vec = cur.fetchone() # Tuple of all columns in that row
        return vec

    def close(self):
        self.conn.commit()
        self.cur.close()
        self.conn.close()




if __name__ == "__main__":
    s = SQLCon()
    print(s.read('king'))
