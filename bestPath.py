from PIL import Image, ImageDraw
import math
import heapq
import sys

"""
@author Chaitanya Patel
"""

# Factor of slowing speed by terrain
terrain_speeds = {
    "orange": 0.8,  # Open land
    "yellow": 2.0,  # Rough meadow
    "white": 1.0,  # Easy movement forest
    "lgreen": 1.4,  # Slow run forest
    "green": 1.8,  # Walk forest
    "dgreen": 3.0,  # Impassible vegetation
    "dblue": 3.0,  # Lake/Swamp/Marsh
    "brown": 0.1,  # Paved road
    "black": 0.3,  # Footpath
    "pink": 3.0,  # Out of bounds
    "lblue": 1.0,  # ice
    "grey": 2.5  # mud
}
uphill_elevation_factor = 1.5
downhill_elevation_factor = 0.5

class PriorityQueue:
    """

    """
    def __init__(self):
        self.queue = []

    def push(self, element, priority):
        heapq.heappush(self.queue, (priority, element))

    def pop(self):
        return heapq.heappop(self.queue)

    def empty(self):
        return len(self.queue) == 0


def getTerrain(pixel):
    """

    :param pixel:
    :return:
    """
    red, green, blue = pixel
    if red == 248 and green == 148 and blue == 18:
        return "orange"
    if red == 255 and green == 192 and blue == 0:
        return "yellow"
    if red == 255 and green == 255 and blue == 255:
        return "white"
    if red == 2 and green == 208 and blue == 60:
        return "lgreen"
    if red == 2 and green == 136 and blue == 40:
        return "green"
    if red == 5 and green == 73 and blue == 24:
        return "dgreen"
    if red == 0 and green == 0 and blue == 255:
        return "dblue"
    if red == 71 and green == 51 and blue == 3:
        return "brown"
    if red == 0 and green == 0 and blue == 0:
        return "black"
    if red == 205 and green == 0 and blue == 101:
        return "pink"
    # Add conditions for new colors acc to seasons here
    return str(red) + " " + str(green) + " " + str(blue)


def heuristicFunction(neighbor, goal, park_map):
    """

    :param neighbor:
    :param goal:
    :param park_map:
    :return:
    """
    # We are representing nodes as (x, y) i.e (width, height)
    # BUT the park map 2 D matrix accesses elements as height , weight.
    # Hence while reading from map we swap the order of node positions
    goalFeatures = park_map[goal[1]][goal[0]]
    neighborFeatures = park_map[neighbor[1]][neighbor[0]]

    distanceFromGoalIn2D = math.sqrt((goal[0] - neighbor[0]) ** 2 + (goal[1] - neighbor[1]) ** 2)
    differenceFromGoalInElevation = abs(goalFeatures[1] - neighborFeatures[1])

    distanceFromGoalIn3D = math.sqrt(distanceFromGoalIn2D ** 2 + differenceFromGoalInElevation ** 2)
    return distanceFromGoalIn3D  # Straight line distance in a 3D plane


def isAround(node, terrain):
    """

    :param node:
    :param terrain:
    :return:
    """
    neighbors = getNeighbors(node)
    for neighbor in neighbors:
        if neighbor[0] == terrain:
            return True
    return False


def evaluationFunction(node, neighbor, park_map, season):
    """

    :param node:
    :param neighbor:
    :param park_map:
    :param season:
    :return:
    """
    # We are representing nodes as (x, y) i.e (width, height)
    # BUT the park map 2 D matrix accesses elements as height , weight.
    # Hence while reading from map we swap the order of node positions
    cost = 1
    nodeFeatures = park_map[node[1]][node[0]]
    neighborFeatures = park_map[neighbor[1]][neighbor[0]]
    # Condition for Fall season
    if season == "fall" and neighborFeatures[0] == "black" and isAround(neighbor, "white"):
        # Footpath values become 0.6 i.e. almost close to easy movement forest value which is 1
        speedDecrease = terrain_speeds.get(neighborFeatures[0])*2
    else:
        speedDecrease = terrain_speeds.get(neighborFeatures[0])
    if neighborFeatures[1] > nodeFeatures[1]:
        # Up hill
        cost = cost + (speedDecrease ** 2) * uphill_elevation_factor * (
                neighborFeatures[1] - nodeFeatures[1]) * 10
    elif neighborFeatures[1] < nodeFeatures[1]:
        # Down hill
        cost = cost + (speedDecrease ** 2) * downhill_elevation_factor * (
                nodeFeatures[1] - neighborFeatures[1]) * 10
    else:
        # Same elevation
        cost = cost + (speedDecrease ** 2) * 1
    return cost


def buildParkMap(img, file):
    """

    :param img:
    :param file:
    :return:
    """
    park_map = []
    width, height = img.size  # 395, 500

    for h in range(height):
        row = []
        line = file.readline().split()
        for w in range(width):
            pixel = img.getpixel((w, h))
            terrain = getTerrain(pixel[:3])
            elevation = float(line[w])
            col = [terrain, elevation]
            row.append(col)
        park_map.append(row)
    return park_map


def getNeighbors(node):
    """

    :param node:
    :return:
    """
    neighbors = []
    x = node[0]
    y = node[1]
    width = 394
    height = 499
    # check if edge node
    if (x == 0 or x == width) or (y == 0 or y == height):
        # top
        if y == 0:
            # top left
            if x == 0:
                neighbors.append((x + 1, y))  # right
                neighbors.append((x, y + 1))  # bottom
                neighbors.append((x + 1, y + 1))  # bottom right
            # top right
            elif x == width:
                neighbors.append((x - 1, y))  # left
                neighbors.append((x, y + 1))  # bottom
                neighbors.append((x - 1, y + 1))  # bottom left
            # top edge but not corners
            else:
                neighbors.append((x + 1, y))  # right
                neighbors.append((x - 1, y))  # left
                neighbors.append((x, y + 1))  # bottom
                neighbors.append((x + 1, y + 1))  # bottom right
                neighbors.append((x - 1, y + 1))  # bottom left
        # bottom
        elif y == height:
            # bottom left
            if x == 0:
                neighbors.append((x, y - 1))  # top
                neighbors.append((x + 1, y))  # right
                neighbors.append((x + 1, y - 1))  # top right
            # bottom right
            if x == width:
                neighbors.append((x, y - 1))  # top
                neighbors.append((x - 1, y))  # left
                neighbors.append((x - 1, y - 1))  # top left
            # bottom edge but not corners
            else:
                neighbors.append((x, y - 1))  # top
                neighbors.append((x + 1, y))  # right
                neighbors.append((x - 1, y))  # left
                neighbors.append((x + 1, y - 1))  # top right
                neighbors.append((x - 1, y - 1))  # top left
        else:
            # left edge but not corners
            if x == 0:
                neighbors.append((x, y - 1))  # top
                neighbors.append((x, y + 1))  # bottom
                neighbors.append((x + 1, y))  # right
                neighbors.append((x + 1, y - 1))  # top right
                neighbors.append((x + 1, y + 1))  # bottom right
            # right edge but not corners
            if x >= width:
                neighbors.append((x, y - 1))  # top
                neighbors.append((x, y + 1))  # bottom
                neighbors.append((x - 1, y))  # left
                neighbors.append((x - 1, y - 1))  # top left
                neighbors.append((x - 1, y + 1))  # bottom left
    else:
        # non edge nodes
        neighbors.append((x - 1, y))  # left
        neighbors.append((x + 1, y))  # right
        neighbors.append((x, y + 1))  # bottom
        neighbors.append((x, y - 1))  # top
        neighbors.append((x - 1, y - 1))  # top left
        neighbors.append((x + 1, y + 1))  # bottom right
        neighbors.append((x + 1, y - 1))  # top right
        neighbors.append((x - 1, y + 1))  # bottom left

    return neighbors

def drawPath(img, path):
    """

    :param img:
    :param path:
    :return:
    """
    for p in path:
        # Path in red
        img.putpixel(p, (255, 0, 0))

def getPath(parents, goal):
    """

    :param parents:
    :param goal:
    :return:
    """
    path = [goal]
    parent = parents.get(goal)
    while parent is not None:
        path.append(parent)
        parent = parents.get(parent)
    return path[::-1]

def drawCheckpoint(img, checkpoint):
    """

    :param img:
    :param checkpoint:
    :return:
    """
    x = checkpoint[0]
    y = checkpoint[1]
    points = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x - 1, y - 1), (x + 1, y + 1), (x + 1, y - 1),
              (x - 1, y + 1)]
    drawPath(img, points)

def findPath(goal, park_map, start, season):
    """

    :param goal:
    :param park_map:
    :param start:
    :param season:
    :return:
    """
    heap = PriorityQueue()
    heap.push(start, 0)
    parents = {start: None}
    costTillNode = {start: 0}
    while not heap.empty():
        node = heap.pop()

        if node[1] == goal:
            break

        neighbors = getNeighbors(node[1])
        for neighbor in neighbors:
            costToNeighbor = costTillNode[node[1]] + evaluationFunction(node[1], neighbor, park_map, season)
            if neighbor not in costTillNode or costToNeighbor < costTillNode[neighbor]:
                costTillNode[neighbor] = costToNeighbor
                costToGoal = heuristicFunction(neighbor, goal, park_map)
                totalCost = costToNeighbor + costToGoal
                heap.push(neighbor, totalCost)
                parents[neighbor] = node[1]

    print("Checkpoint reached", goal)
    return costTillNode, parents


def getWaterEdges(park_map):
    """

    :param park_map:
    :return:
    """
    water = "dblue"
    leftOfWater = []
    rightOfWater = []
    topOfWater = []
    bottomOfWater = []
    width, height = 394, 499
    for h in range(height):
        for w in range(width):
            if park_map[h][w][0] != water and park_map[h][w+1][0] == water:
                leftOfWater.append((h, w))
            if park_map[h][w][0] != water and park_map[h][w-1][0] == water:
                rightOfWater.append((h, w))
            if park_map[h][w][0] != water and park_map[h-1][w][0] == water:
                topOfWater.append((h, w))
            if park_map[h][w][0] != water and park_map[h+1][w][0] == water:
                bottomOfWater.append((h, w))
    return leftOfWater, rightOfWater, topOfWater, bottomOfWater


def getIceNodes(park_map, edgeNodes, extent, direction):
    """

    :param park_map:
    :param edgeNodes:
    :param extent:
    :param direction:
    :return:
    """
    originalTerrain = "dblue"
    affectedNodes = []
    for edge in edgeNodes:
        if direction == "right" and edge[1] + extent < 395 and edge[0] < 500:
            for i in range(1, extent + 1):
                if park_map[edge[0]][edge[1] + i][0] is originalTerrain:
                    affectedNodes.append((edge[0], edge[1]+i))
        if direction == "left" and edge[1] - extent < 395 and edge[0] < 500:
            for i in range(1, extent + 1):
                if park_map[edge[0]][edge[1] - i][0] is originalTerrain:
                    affectedNodes.append((edge[0], edge[1]-i))
        if direction == "down" and edge[1] < 395 and edge[0] + extent < 500:
            for i in range(1, extent + 1):
                if park_map[edge[0] + i][edge[1]][0] is originalTerrain:
                    affectedNodes.append((edge[0]+i, edge[1]))
        if direction == "up" and edge[1] < 395 and edge[0] - extent < 500:
            for i in range(1, extent + 1):
                if park_map[edge[0] - i][edge[1]][0] is originalTerrain:
                    affectedNodes.append((edge[0]-i, edge[1]))

    return affectedNodes

def getMudNodes(park_map, edgeNodes, extent, direction):
    """

    :param park_map:
    :param edgeNodes:
    :param extent:
    :param direction:
    :return:
    """
    water = "dblue"
    affectedNodes = []
    for edge in edgeNodes:
        if direction == "right" and edge[1] + extent < 395 and edge[0] < 500:
            waterNodeElevation = park_map[edge[0]][edge[1] - 1][1]
            for i in range(0, extent + 1):
                landNode = park_map[edge[0]][edge[1] + i]
                if landNode[1]-waterNodeElevation < 1.0 and landNode[0] != water:
                    affectedNodes.append((edge[0], edge[1]+i))
        if direction == "left" and edge[1] - extent < 395 and edge[0] < 500:
            waterNodeElevation = park_map[edge[0]][edge[1] + 1][1]
            for i in range(0, extent + 1):
                landNode = park_map[edge[0]][edge[1] - i]
                if landNode[1] - waterNodeElevation < 1.0 and landNode[0] != water:
                    affectedNodes.append((edge[0], edge[1]-i))
        if direction == "down" and edge[1] < 395 and edge[0]+extent < 500:
            waterNodeElevation = park_map[edge[0] - 1][edge[1]][1]
            for i in range(0, extent + 1):
                landNode = park_map[edge[0] + i][edge[1]]
                if landNode[1] - waterNodeElevation < 1.0 and landNode[0] != water:
                    affectedNodes.append((edge[0]+i, edge[1]))
        if direction == "up" and edge[1] < 395 and edge[0]-extent < 500:
            waterNodeElevation = park_map[edge[0] + 1][edge[1]][1]
            for i in range(0, extent + 1):
                landNode = park_map[edge[0] - i][edge[1]]
                if landNode[1] - waterNodeElevation < 1.0 and landNode[0] != water:
                    affectedNodes.append((edge[0]-i, edge[1]))

    return affectedNodes

def effectsOfWinter(img, park_map):
    """

    :param img:
    :param park_map:
    :return:
    """
    # Find water edges in 4 directions
    leftOfWater, rightOfWater, topOfWater, bottomOfWater = getWaterEdges(park_map)
    # Add nodes 7 spaces in that direction to a set
    ice = set()
    ice.update(getIceNodes(park_map, leftOfWater, 7, "right"))
    ice.update(getIceNodes(park_map, rightOfWater, 7, "left"))
    ice.update(getIceNodes(park_map, topOfWater, 7, "down"))
    ice.update(getIceNodes(park_map, bottomOfWater, 7, "up"))
    # Change color value in image file
    #  & Update color in park_map for these node
    img2 = img.copy()
    # Using updated image file
    for ic in ice:
        park_map[ic[0]][ic[1]][0] = "lblue"
        img2.putpixel((ic[1], ic[0]), (110, 255, 255))
    return img2


def effectsOfSpring(img, park_map):
    """

    :param img:
    :param park_map:
    :return:
    """
    leftOfWater, rightOfWater, topOfWater, bottomOfWater = getWaterEdges(park_map)
    mud = set()
    mud.update(getMudNodes(park_map, leftOfWater, 15, "left"))
    mud.update(getMudNodes(park_map, rightOfWater, 7, "right"))
    mud.update(getMudNodes(park_map, topOfWater, 7, "up"))
    mud.update(getMudNodes(park_map, bottomOfWater, 7, "down"))
    img2 = img.copy()
    for m in mud:
        park_map[m[0]][m[1]][0] = "grey"
        img2.putpixel((m[1], m[0]), (160, 160, 160))
    return img2


def main():
    """

    :return:
    """
    # terrain - image, elevation - file, path - file,
    # season(summer, fall, winter, or spring), output - image - filename.
    img = Image.open(sys.argv[1])
    elevationFile = open(sys.argv[2])
    park_map = buildParkMap(img, elevationFile)

    pathFile = open(sys.argv[3])
    checkpoints = []
    for line in pathFile:
        coordinates = line.split()
        checkpoints.append((int(coordinates[0]), int(coordinates[1])))
    start = checkpoints[0]

    season = sys.argv[4].lower()
    if season == "winter":
        img2 = effectsOfWinter(img, park_map)
    elif season == "spring":
        img2 = effectsOfSpring(img, park_map)
    else:
        # Summer and Fall
        img2 = img.copy()

    outputImageName = sys.argv[5]

    print("Starting run...", start)
    totalCost = 0
    for g in checkpoints[1:]:
        goal = (int(g[0]), int(g[1]))
        costTillNode, parents = findPath(goal, park_map, start, season)
        path = getPath(parents, goal)
        drawPath(img2, path)
        print("Path:", path)
        print("Cost to checkpoint:", goal, costTillNode[goal])
        totalCost += costTillNode[goal]
        start = goal
        drawCheckpoint(img2, start)

    print("Total cost of path:", totalCost)
    draw2 = ImageDraw.Draw(img2)
    draw2.text((0, 0), text="Cost: "+str(totalCost), fill=(0, 0, 0))
    img2.show()
    img2.save(outputImageName)


if __name__ == '__main__':
    main()
