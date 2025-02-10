import numpy as np
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, maxDisappeared=40):
        # Initialize the next object ID, dictionaries for object centroids and
        # the number of consecutive frames an object has not been seen.
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # Assign a new object ID and initialize its centroid and disappearance count.
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Remove an object ID from tracking.
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """
        Update the tracker with a list of bounding boxes (each box is (startX, startY, endX, endY)).
        Returns the dictionary mapping object IDs to centroids.
        """
        if len(rects) == 0:
            # No detections: mark existing objects as disappeared.
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Compute the centroids for the new bounding boxes.
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If no objects are currently tracked, register all centroids.
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # Grab the set of object IDs and their centroids.
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute the distance matrix between existing centroids and new centroids.
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # Find the smallest value in each row and then sort the row indexes based on their minimum value.
            rows = D.min(axis=1).argsort()
            # Then, find the corresponding column index for each row.
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # Loop over the combination of (row, column) index.
            for (row, col) in zip(rows, cols):
                # If we have already examined either the row or column, skip.
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            # Compute row and column indices we have NOT examined.
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # If the number of object centroids is equal to or greater than the input centroids,
            # mark the corresponding objects as disappeared.
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                # Otherwise, register each new input centroid as a new object.
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects