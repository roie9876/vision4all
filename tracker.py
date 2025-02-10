import numpy as np
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.nextObjectID = 0
        self.objects = {}       # objectID -> centroid
        self.disappeared = {}   # objectID -> number of consecutive missing frames
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # rects: list of bounding boxes in format (x, y, w, h)
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        input_centroids = []
        for (x, y, w, h) in rects:
            cX = int(x + w / 2)
            cY = int(y + h / 2)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), np.array(input_centroids))
            rows = D.min(axis=1).argsort()
            usedCols = set()
            for row in rows:
                col = D[row].argmin()
                if col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = input_centroids[col]
                self.disappeared[objectID] = 0
                usedCols.add(col)
            for i, centroid in enumerate(input_centroids):
                if i not in usedCols:
                    self.register(centroid)
        return self.objects
