import math


class EuclideanDistTracker:
    def __init__(self, lookback):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the
        # unt of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

        # Center positions which have been lost, to be compared with when a new object is found
        self.historic_points = {}

        # History backlog aggression
        self.lookback = lookback




    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            historic_object_detected = False
            same_object_detected = False


            if len(self.historic_points) > 0:
                for id in list(reversed(list(self.historic_points)))[0:self.lookback]:
                    pt = self.historic_points.get(id)
                    dist = math.hypot(cx - pt[0], cy - pt[1])
                    if dist < 10:
                        self.center_points[id] = (cx, cy)
                        print(f"Log at {cx}:{cy} could be {id}")
                        break


            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 150:
                    self.center_points[id] = (cx, cy)
                    self.historic_points[id] =(cx, cy)
                    # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break






            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids



