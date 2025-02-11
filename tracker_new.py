import numpy as np
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        # המזהה הבא שיוקצה לעצם חדש
        self.nextObjectID = 0
        # מילון לשמירת העצמים במעקב (מזהה: מרכז)
        self.objects = {}
        # מילון למעקב אחרי מספר הפריימים שבהם העצם לא זוהה
        self.disappeared = {}
        # מספר הפריימים המקסימלי שבו ניתן לראות עצם לפני שנמחקו
        self.max_disappeared = max_disappeared
        # מרחק מקסימלי בין המרכזים על מנת להתאים בין עצמים קיימים לגילויים חדשים
        self.max_distance = max_distance

    def register(self, centroid):
        # רישום עצם חדש עם המרכז הנתון
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # הסרת עצם מהמעקב
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """
        מקבל רשימה של תיבות גבול (בפורמט x, y, w, h) מהזיהויים הנוכחיים.
        מחשב את מרכזי התיבות ומעדכן את המעקב:
         - אם אין זיהויים, מגדיל את מספר הפריימים בהם העצם נעדר.
         - אם יש זיהויים, מתאם ביניהם לבין העצמים הקיימים לפי מרחק.
         - רושם מופעים חדשים אם נדרש.
        מחזיר את רשימת העצמים עם המרכזים המעודכנים.
        """
        # אם אין זיהויים נוכחיים, עדכן את כל העצמים כנעדרים
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        # חשב את מרכזי התיבות של הזיהויים הנוכחיים
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            inputCentroids[i] = (cX, cY)

        # אם אין עצמים במעקב, רשם את כל הזיהויים כמקרים חדשים
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # רשום את העצמים הקיימים ואת המרכזים שלהם
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # חשב את המטריצה של מרחקים בין המרכזים הקיימים למרכזי הזיהויים
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # סדר את השורות (עצמים קיימים) לפי הערך המינימלי בכל שורה
            rows = D.min(axis=1).argsort()

            # עבור כל שורה, קבל את האינדקס של המרכז הקרוב ביותר בזיהויים
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # התאמה בין העצמים הקיימים לזיהויים חדשים
            for (row, col) in zip(rows, cols):
                # אם כבר השתמשנו בשורה או בעמודה הזו, דלג
                if row in usedRows or col in usedCols:
                    continue

                # אם המרחק גדול מדי, דלג (אין התאמה)
                if D[row, col] > self.max_distance:
                    continue

                # עדכן את העצם עם המרכז החדש
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            # מצא את השורות והעמודות שלא הותאמו
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # אם מספר העצמים הקיימים גדול או שווה למספר הזיהויים, עדכן את העצמים הנעדרים
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)
            else:
                # אחרת, רשום את הזיהויים החדשים כעצמים חדשים
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # החזר את רשימת העצמים המעודכנים
        return self.objects