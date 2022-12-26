import pygame, sys
import math
import numpy as np
from pygame.locals import *
import network

def simulate(candidates):
    pygame.init()
    w,h = 640, 360
    screen = pygame.display.set_mode((w,h))
    pygame.display.set_caption("environment.py")
    track_img = pygame.image.load("track.png").convert_alpha()
    car_img = pygame.image.load("car.png")
    activecar_img = pygame.image.load("activecar.png")
    inactivecar_img = pygame.image.load("inactivecar.png")
    #track = pygame.Surface((1952, 1824), pygame.SRCALPHA)

    class Car():
        def __init__(self, position, weights, bias):
            self.posX, self.posY = position[0], position[1]

            self.weights = weights
            self.bias = bias
            self.network = network.Network(self.weights, self.bias)

            self.image = pygame.Surface((15,30), pygame.SRCALPHA)
            self.image.blit(car_img,(0,0))
            self.original_image = self.image
            self.rect = self.image.get_rect(center=(self.posX-cameraX, self.posY-cameraY))

            self.angle = 0
            self.leftsensor_angle = math.pi/4
            self.rightsensor_angle = -math.pi/4
            self.angle = 0

            self.sensors = {
                "left_sensor": 0,
                "front_sensor": 0,
                "right_sensor": 0
            }

            self.acceleration = 0.1
            self.velocity = 0.1

            self.freeze = False

            self.fitness = 1
            self.old_fitness = -1

            self.last_colour = (88,232,88,255)

        def move(self):
            if not self.freeze:
                self.posX += math.sin(self.angle) * -self.velocity
                self.posY += math.cos(self.angle) * -self.velocity

        def rotate(self, direction):
            if not self.freeze:
                if direction == "left":
                    self.angle += 0.3#math.pi/10#/45
                    self.leftsensor_angle += 0.3#math.pi/10#/45
                    self.rightsensor_angle += 0.3#math.pi/10#/45 #pi/360
                elif direction == "right":
                    self.angle -= 0.3#math.pi/10#/45
                    self.leftsensor_angle -= 0.3#math.pi/10#/45
                    self.rightsensor_angle -= 0.3#math.pi/10#/45
                self.angle %= 2*math.pi
                self.leftsensor_angle %= 2*math.pi
                self.rightsensor_angle %= 2*math.pi
                self.image = pygame.transform.rotate(self.original_image, math.degrees(self.angle))#, 1)
                self.rect = self.image.get_rect(center=(self.posX, self.posY))#center=self.rect.center)

        def kill(self):
            self.freeze = True
            self.image = pygame.transform.rotate(self.original_image, math.degrees(0))
            self.image.blit(inactivecar_img,(0,0))
            self.original_image = self.image
            self.image = pygame.transform.rotate(self.original_image, math.degrees(self.angle))

        def draw(self):
            screen.blit(self.image, (self.posX-cameraX, self.posY-cameraY))

        def update_fitness(self):
            if not self.freeze:
                colours = [(254,246,246,255), (247, 253, 247, 255), (253, 253, 246, 255)]#[(253,240,240,255), (241,253,241,255), (254,253,241,255)]
                for i in range(30):
                    for j in range(30):
                        try:
                            if self.image.get_at((i,j)) in [(237, 28, 36, 255), (34, 177, 76, 255)]:
                                colour = track_img.get_at((int(self.posX+i), int(self.posY+j)))
                                if colour in colours: #change
                                    if colour != self.last_colour:
                                        if self.last_colour == colours[0] and colour == colours[1]:
                                            self.fitness += 1
                                        elif self.last_colour == colours[1] and colour == colours[2]:
                                            self.fitness += 1
                                        elif self.last_colour == colours[2] and colour == colours[0]:
                                            self.fitness += 1
                                        elif self.last_colour == colours[1] and colour == colours[0]:
                                            self.fitness -= 1
                                        elif self.last_colour == colours[2] and colour == colours[1]:
                                            self.fitness -= 1
                                        elif self.last_colour == colours[0] and colour == colours[2]:
                                            self.fitness -= 1
                                    self.last_colour = colour
                        except IndexError:
                            pass

        def draw_sensors(self):
            if not self.freeze:
                sensors = {
                    "left_sensor": {"front":None, "back":None, "colour": (38, 173, 20, 255), "angle": self.leftsensor_angle},
                    "front_sensor": {"front":None, "back":None, "colour": (0, 104, 196, 255), "angle": self.angle},
                    "right_sensor": {"front":None, "back":None, "colour": (196, 177, 23, 255), "angle": self.rightsensor_angle}
                }
                for i in range(30):
                    for j in range(30):
                        if (sensors["left_sensor"]["front"] and sensors["left_sensor"]["back"] and sensors["front_sensor"]["front"] and sensors["front_sensor"]["back"] and sensors["right_sensor"]["front"] and sensors["right_sensor"]["back"]):
                            break
                        try:
                            colour = self.image.get_at((i,j))
                            if colour == (38, 173, 20, 255):
                                sensors["left_sensor"]["front"] = (i,j)
                            elif colour == (52, 237, 28, 255):
                                sensors["left_sensor"]["back"] = (i,j)
                            elif colour == (0, 104, 196, 255):
                                sensors["front_sensor"]["front"] = (i,j)
                            elif colour == (0, 136, 255, 255):
                                sensors["front_sensor"]["back"] = (i,j)
                            elif colour == (196, 177, 23, 255):
                                sensors["right_sensor"]["front"] = (i,j)
                            elif colour == (237, 215, 28, 255):
                                sensors["right_sensor"]["back"] = (i,j)
                        except IndexError:
                            pass
                    if (sensors["left_sensor"]["front"] and sensors["left_sensor"]["back"] and sensors["front_sensor"]["front"] and sensors["front_sensor"]["back"] and sensors["right_sensor"]["front"] and sensors["right_sensor"]["back"]):
                        break
                for key, sensor in sensors.items():
                    if (sensor["front"] and sensor["back"]):
                        i = 0
                        touchingWall = False
                        current_pos = (self.posX+sensor["front"][0]-cameraX, self.posY+sensor["front"][1]-cameraY)
                        draw_coordinates = []
                        while (not touchingWall) and i < 100:
                            i+=1
                            current_pos = (current_pos[0]-(1*math.sin(sensor["angle"])), current_pos[1]-(math.cos(sensor["angle"])*1))
                            int_current_pos = [int(round(current_pos[0]+cameraX)), int(round(current_pos[1]+cameraY))]
                            try:
                                if track_img.get_at(int_current_pos) == (0,0,0,255):
                                    touchingWall = True
                                else:
                                    draw_coordinates.append(current_pos)
                            except IndexError:
                                pass
                        for pos in draw_coordinates:
                            screen.fill(sensor["colour"], (pos, (2, 2)))
                        self.sensors[key] = i
    def check_collision(car):
        if not car.freeze:
            for i in range(30):
                for j in range(30):
                    try:
                        if car.image.get_at((i,j)) in [(237, 28, 36, 255), (34, 177, 76, 255)]:
                            if track_img.get_at((int(car.posX+i), int(car.posY+j))) == (0,0,0,255):
                                car.kill()
                                break
                    except IndexError:
                        pass
                if car.freeze:
                    break

    cameraX, cameraY = 0,0

    cars = []
    for car in candidates:
        layers = [3,4,4,2]
        cars.append(Car([190,920], car.weights, car.bias)) #167,1410

    active_car = cars[0]

    clock = pygame.time.Clock()
    ticks = 0
    while True: #Main game loop
        ticks += 1
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        if not active_car.freeze:
            active_car.image = pygame.transform.rotate(active_car.original_image, math.degrees(0))
            active_car.image.blit(car_img,(0,0))
            active_car.original_image = active_car.image
            active_car.image = pygame.transform.rotate(active_car.original_image, math.degrees(active_car.angle))
        for car in cars:
            if (active_car.freeze and not car.freeze) or (car.fitness > active_car.fitness and not car.freeze):
                active_car = car
        if not active_car.freeze:
            active_car.image = pygame.transform.rotate(active_car.original_image, math.degrees(0))
            active_car.image.blit(activecar_img,(0,0))
            active_car.original_image = active_car.image
            active_car.image = pygame.transform.rotate(active_car.original_image, math.degrees(active_car.angle))

        for car in cars: #Update velocity and rotation from feed-forward ouput
            output = car.network.forward_propagate(np.asmatrix([car.sensors["left_sensor"],car.sensors["front_sensor"],car.sensors["right_sensor"]]).transpose())
            if output[0] >= 0.66:
                if car.velocity < 15:
                    car.velocity += car.acceleration
            elif  output[1] >= 0.33:
                if car.velocity > 0.1:
                    car.velocity -= (car.acceleration)
                else:
                    car.velocity = 0
            if output[1] >= 0.66:
                car.rotate("left")
            elif output[1] >= 0.33:
                car.rotate("right")

        screen.fill((255,255,255))
        cameraX = active_car.posX - 0.5*w
        cameraY = active_car.posY - 0.5*h
        screen.blit(track_img,(0 -cameraX,0 -cameraY))
        for car in cars:
            if not car.freeze:
                if ticks % 150 == 0:
                    if car.fitness == car.old_fitness:
                        car.kill()
                    else:
                        car.old_fitness = car.fitness
            car.move()
            car.draw()
            car.draw_sensors()
            car.update_fitness()
            check_collision(car)

        pygame.display.flip()
        clock.tick(90)
        if ticks % 200 == 0:
            for car in cars:
                if not car.freeze:
                    break
            else:
                pygame.quit()
                for car in cars:
                    if car.fitness < 1: car.fitness = 1
                    #print(car.fitness)
                return cars
