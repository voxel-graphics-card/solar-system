#include <SDL3/SDL.h>               // Include SDL3 header for graphics
#include <SDL3_ttf/SDL_ttf.h>         // Include SDL_ttf for text rendering
#include <cmath>                     // Include cmath for math functions (sqrt, sin, cos)
#include <cstdio>                    // Standard I/O operations
#include <cstdlib>                   // For memory allocation and other utilities
#include <iostream>                  // For C++ I/O streams (cout, cerr)
#include <vector>                    // For dynamic arrays (std::vector)
#include <sstream>                   // For converting numbers to strings
#define M_PI 3.14159265358979323846 // Define the value of PI if not already defined



// Utility functions to convert an int or float into a string using a string stream.
std::string to_string(int value) {
    std::ostringstream ss;         // Create a stream object for output
    ss << value;                   // Insert the integer into the stream
    return ss.str();               // Return the string from the stream
}

std::string to_string(float value) {
    std::ostringstream ss;         // Create a stream object for output
    ss << value;                   // Insert the float value into the stream
    return ss.str();               // Return the resulting string
}

// Global gravitational constant (G) used in the simulation; can be modified via keyboard.
float G = 8.0f;

// 2D vector struct for positions, velocities, and other 2D computations.
struct Vec2 {
    float x, y;                    // Components of the vector in the x and y directions.
};

// Operator overloads for basic vector operations:

// Vector addition.
Vec2 operator+(const Vec2 &a, const Vec2 &b) {
    return { a.x + b.x, a.y + b.y };
}

// Vector subtraction.
Vec2 operator-(const Vec2 &a, const Vec2 &b) {
    return { a.x - b.x, a.y - b.y };
}

// Scalar multiplication: scale a vector by a float.
Vec2 operator*(const Vec2 &v, float s) {
    return { v.x * s, v.y * s };
}

// Scalar division: divide a vector by a float.
Vec2 operator/(const Vec2 &v, float s) {
    return { v.x / s, v.y / s };
}

// Return the length (magnitude) of a vector.
float length(const Vec2 &v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

// Return a normalized (unit length) version of a vector.
Vec2 normalize(const Vec2 &v) {
    float len = length(v);
    // If length is positive, return the vector divided by its length; otherwise, return a zero vector.
    return (len > 0 ? v / len : Vec2{0, 0});
}

//------------------------------------------------------------------------------
// Class representing a celestial body in the simulation (e.g. a planet or star).
//------------------------------------------------------------------------------
class Body {
public:
    float mass;                 // Mass of the body (affects gravitational force).
    Vec2 pos;                   // Position (in world coordinates).
    Vec2 vel;                   // Velocity vector.
    bool isVisible;               //is body visible on screen?
    SDL_Color color;            // Color for rendering the body.
    float radius;               // Radius of the body (used for drawing and collision approximations).
    std::vector<Vec2> trail;    // History of positions to draw motion trails.
    size_t maxTrailLength = 1000;// Maximum number of positions stored in the trail.

    // Constructor to initialize the body with its mass, position, velocity, color, and radius respectivly...
    Body(float m, Vec2 p, Vec2 v, SDL_Color col, float r)
        : mass(m), pos(p), vel(v), color(col), radius(r) {}

};

//------------------------------------------------------------------------------
// Global variables for view transformation (panning and zooming).
//------------------------------------------------------------------------------
float zoom = 1.0f;          // Zoom factor for scaling world coordinates to screen coordinates.
Vec2 offset = {0, 0};       // Offset(initial position) for panning the view (affects the "center" of the view).

//------------------------------------------------------------------------------
// Coordinate conversion functions:
//------------------------------------------------------------------------------

// Convert a point in world coordinates to screen coordinates for rendering.
// It centers the world in the window and applies zoom and offset.
SDL_FPoint toScreen(const Vec2& pos, int winWidth, int winHeight) {
    return {
        winWidth / 2 + (pos.x + offset.x) * zoom,    // X coordinate: center + (world_x + offset) * zoom.
        winHeight / 2 - (pos.y + offset.y) * zoom     // Y coordinate: center - (world_y + offset) * zoom (flip y-axis).
    };
}

// Convert from screen coordinates back to world coordinates.
// This is useful when handling mouse input.
Vec2 screenToWorld(int screenX, int screenY, int winWidth, int winHeight) {
    return {
        (screenX - winWidth / 2.0f) / zoom - offset.x, // Reverse the horizontal transformation.
        -(screenY - winHeight / 2.0f) / zoom - offset.y // Reverse the vertical transformation (accounting for axis flip).
    };
}

//------------------------------------------------------------------------------
// Rendering Functions:
//------------------------------------------------------------------------------

// Draw the trail (history of positions) for a given body.
// For each position stored in the trail, convert from world to screen coordinates and draw a point.
void drawTrail(SDL_Renderer* renderer, const Body* body, int winWidth, int winHeight) {
    SDL_SetRenderDrawColor(renderer, body->color.r, body->color.g, body->color.b, 80);
    for (const Vec2& p : body->trail) {
        SDL_FPoint sp = toScreen(p, winWidth, winHeight);
        SDL_RenderPoint(renderer, static_cast<int>(sp.x), static_cast<int>(sp.y));
    }
}

#include <algorithm> // Include the <algorithm> header for std::remove_if
#include <memory> // Include memory for std::unique_ptr

//------------------------------------------------------------------------------
// Function to cull bodies that are outside the visible screen area.
//------------------------------------------------------------------------------
void cullBodies(std::vector<std::unique_ptr<Body>>& bodies, int winWidth, int winHeight) {
    // Calculate the screen boundaries in world coordinates.
    Vec2 worldTopLeft = screenToWorld(0, 0, winWidth, winHeight);
    Vec2 worldBottomRight = screenToWorld(winWidth, winHeight, winWidth, winHeight);

    // Define the bounding box for the visible area in world coordinates.
    float minX = worldTopLeft.x;
    float maxX = worldBottomRight.x;
    float minY = worldBottomRight.y; // Corrected to use bottom right for minY
    float maxY = worldTopLeft.y; // Corrected to use top left for maxY

    // Use a margin to account for the maximum radius of bodies.
    const float margin = 100.0f; // A fixed margin to ensure partial visibility is considered.

    // Adjust the bounding box with the margin.
    minX -= margin;
    minY -= margin;
    maxX += margin;
    maxY += margin;

    // Debugging output for bounding box
    //std::cout << "Bounding box: (" << minX << ", " << minY << ") - (" << maxX << ", " << maxY << ")" << std::endl;

    // Update visibility flags instead of removing bodies
    for (auto& body : bodies) {
        body->isVisible = !(body->pos.x + body->radius < minX ||
                            body->pos.x - body->radius > maxX ||
                            body->pos.y + body->radius < minY ||
                            body->pos.y - body->radius > maxY);
    }
}



// Draw the velocity vector of a body as a line.
// The length of the drawn line is scaled to represent the velocity magnitude.
void drawVelocityVector(SDL_Renderer* renderer, const Vec2& position, const Vec2& velocity, int winWidth, int winHeight) {
    const float scale = 0.5f;   // Scaling factor for the velocity vector so it is visually reasonable.
    SDL_FPoint bodyt = toScreen(position, winWidth, winHeight);               // Convert the body's position to screen coordinates.
    SDL_FPoint end = toScreen(position + (velocity * scale), winWidth, winHeight); // Calculate and convert the end of the velocity vector.
    SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255);   // Use yellow color for the velocity vector.
    SDL_RenderLine(renderer, static_cast<int>(bodyt.x), static_cast<int>(bodyt.y),
                   static_cast<int>(end.x), static_cast<int>(end.y));  // Draw the line.
}

// Draw a filled circle to represent a body.
// This naïve implementation draws a pixel at every point inside the circle's radius.
void drawFilledCircle(SDL_Renderer* renderer, float cx, float cy, float radius, SDL_Color color) {
    // Set the drawing color according to the specified color.
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    int r = static_cast<int>(radius); // Convert radius to an integer.
    // Loop over a square region covering the circle.
    for (int w = -r; w <= r; w++) {
        for (int h = -r; h <= r; h++) {
            // If the point is inside the circle (circle equation: x^2 + y^2 <= radius^2)
            if ((w * w + h * h) <= (radius * radius)) {
                // Draw the point at the corresponding position.
                SDL_RenderPoint(renderer, static_cast<int>(cx + w), static_cast<int>(cy + h));
            }
        }
    }
}


//collision detection
//idk man why the hell did i implement it sooooo late???
// This function processes collisions between bodies iteratively. If two bodies overlap,
// and if at least one is heavy enough, they merge. Otherwise, if they overlap slightly,
// a gentle repulsive force is applied to push them apart.
void handleCollisions(std::vector<std::unique_ptr<Body>>& bodies) {
    const float collisionBuffer = 0.05f;
    const int maxIterations = 250;
    int iterations = 0;
    bool collisionDetected = true;

    while (collisionDetected && iterations < maxIterations) {
        collisionDetected = false;

        for (size_t i = 0; i < bodies.size(); ++i) {
            for (size_t j = i + 1; j < bodies.size(); ++j) {
                Body* a = bodies[i].get();
                Body* b = bodies[j].get();

                Vec2 d = b->pos - a->pos;
                float dist = length(d);
                if (dist < 0.001f)
                    dist = 0.001f;
                Vec2 dir = d / dist;
                float rSum = a->radius + b->radius;

                if (dist < (rSum + collisionBuffer) && (a->mass > 100000 || b->mass > 100000)) {
                    float totalMass = a->mass + b->mass;
                    Vec2 newPos = (a->pos * a->mass + b->pos * b->mass) / totalMass;
                    Vec2 newVel = (a->vel * a->mass + b->vel * b->mass) / totalMass;
                    SDL_Color newColor = {
                        static_cast<Uint8>((a->color.r + b->color.r) / 2),
                        static_cast<Uint8>((a->color.g + b->color.g) / 2),
                        static_cast<Uint8>((a->color.b + b->color.b) / 2),
                        255
                    };
                    float newRadius = std::sqrt(a->radius * a->radius + b->radius * b->radius);
                    auto merged = std::make_unique<Body>(totalMass, newPos, newVel, newColor, newRadius);
                    merged->trail = a->trail;

                    bodies.erase(bodies.begin() + j);
                    bodies.erase(bodies.begin() + i);
                    bodies.push_back(std::move(merged));

                    collisionDetected = true;
                    goto nextIteration;
                }
                else if (dist < (rSum + collisionBuffer)) {
                    float overlap = (rSum + collisionBuffer) - dist;
                    overlap = std::max(0.0f, overlap);

                    float repulsionStrength = 2.0f;
                    float repulseMag = std::min(overlap * repulsionStrength, 10.0f);
                    Vec2 repulse = dir * repulseMag;

                    float totalMass = a->mass + b->mass;
                    float aFactor = b->mass / totalMass;
                    float bFactor = a->mass / totalMass;

                    a->vel = a->vel - repulse * aFactor;
                    b->vel = b->vel + repulse * bFactor;

                    collisionDetected = true;
                    goto nextIteration;
                }
            }
        }
        nextIteration:
        iterations++;
    }
}




// Draw light rays emanating from the brightest star.
// Each ray is drawn from the star until it intersects another body (if it does).
void drawLightRays(SDL_Renderer* renderer, std::vector<std::unique_ptr<Body>>& bodies, int winWidth, int winHeight) {
    // Identify the "star" (assumed to be the light source) as the body with the largest mass.
    const Body* star = nullptr;
    for (const auto& bodyPtr : bodies) {
        const Body* body = bodyPtr.get(); // Dereference the unique_ptr to get the raw pointer
        if (!star || body->mass > star->mass) {
            star = body;
        }
    }
    // If no star is found, do nothing.
    if (!star) return;

    // Convert the star's position to screen coordinates (for possible use in rendering).
    SDL_FPoint starScreen = toScreen(star->pos, winWidth, winHeight);

    const int rayCount = 360;       // Total number of rays to draw in a full circle.
    const float rayLength = 600.0f;   // Length (in world units) to which each ray extends.
    SDL_SetRenderDrawColor(renderer, star->color.r, star->color.g, star->color.b, 30); // Set a light of the color of star with transparency for the rays.

    // Loop through and draw each ray.
    for (int i = 0; i < rayCount; ++i) {
        // Determine the angle for this ray to evenly cover 360 degrees.
        float angle = i * 2 * M_PI / rayCount;
        Vec2 dir = { std::cos(angle), std::sin(angle) }; // Unit vector in the ray's direction.
        Vec2 rayStart = star->pos;                // All rays originate at the star's position.
        Vec2 rayEnd = star->pos + dir * rayLength;  // Compute the default end point of the ray.

        // Check if the ray intersects any other body.
        for (const auto& bodyPtr : bodies) {
            const Body* target = bodyPtr.get(); // Dereference the unique_ptr to get the raw pointer
            if (target == star) continue; // Skip the star itself.
            Vec2 toTarget = target->pos - rayStart; // Distance of target from center of the star
            // Project the target's position onto the ray direction to find the closest approach.
            float proj = (toTarget.x * dir.x + toTarget.y * dir.y);
            // Clamp the projection value to the ray's length.
            proj = std::fmax(0, std::fmin(rayLength, proj));
            Vec2 closestPoint = rayStart + dir * proj;  // Closest point on the ray to the target.
            float dist = length(closestPoint - target->pos);  // Distance from the target to the ray.
            // If this distance is less than the target's radius, the ray hits the target.
            if (dist < target->radius) {
                rayEnd = closestPoint; // Shorten the ray to the point of intersection.
                break;
            }
        }
        // Convert the ray start and end points from world to screen coordinates.
        SDL_FPoint screenStart = toScreen(rayStart, winWidth, winHeight);
        SDL_FPoint screenEnd = toScreen(rayEnd, winWidth, winHeight);
        // Draw the ray as a line on the screen.
        SDL_RenderLine(renderer, static_cast<int>(screenStart.x), static_cast<int>(screenStart.y),
                                  static_cast<int>(screenEnd.x), static_cast<int>(screenEnd.y));
    }
}



void resetSimulation(std::vector<std::unique_ptr<Body>>& bodies) {
    bodies.clear();

    bodies.push_back(std::make_unique<Body>(100000, Vec2{0, 0}, Vec2{0, 0}, SDL_Color{255, 255, 0, 255}, 20));
    bodies.push_back(std::make_unique<Body>(500, Vec2{200, 0}, Vec2{0, 70}, SDL_Color{0, 150, 255, 255}, 5));
    bodies.push_back(std::make_unique<Body>(500, Vec2{250, 0}, Vec2{0, 85}, SDL_Color{255, 100, 100, 255}, 3));
}








//-----------------------------------------------------------------------------
//old euler method
//-----------------------------------------------------------------------------
// void updateEuler(const std::vector<Body*>& others, float dt) {
//     Vec2 a = {0, 0};           // Start with zero acceleration.
//     for (auto other : others) {
//         if (other == this) continue; // Skip self.
//         Vec2 d = other->pos - pos;   // Difference vector from self to other.
//         float dist = length(d);      // Compute the distance magnitude.
//         const float softening = 10.0f; // Softening factor prevents extreme acceleration at small distances.
//         float denom = (dist * dist) + (softening * softening);
//         float accelMag = G * other->mass / denom;  // Compute the gravitational acceleration magnitude.
//         a = a + normalize(d) * accelMag;           // Add the acceleration vector (in the direction of other).
//     }
//     vel = vel + a * dt;        // Update velocity: new velocity = old velocity + acceleration * dt.
//     pos = pos + vel * dt;      // Update position: new position = old position + velocity * dt.
//     trail.push_back(pos);      // Append new position to the movement trail.
//     if (trail.size() > maxTrailLength) {
//         trail.erase(trail.begin());  // Trim trail to max length.
//     }
// }




//------------------------------------------------------------------------------
// Runge–Kutta (RK4) Integration for Improved Simulation Accuracy
//------------------------------------------------------------------------------

// Structure to hold the state (position and velocity) of a body.
struct State {
    Vec2 pos;
    Vec2 vel;
};



//
// computeAcceleration:
// For a given body indexed by 'i', calculate the gravitational acceleration
// acting on it using the provided vector of states.
// The bodies' masses are taken from the actual Body objects, while positions are
// from the temporary state vector 'states' (allowing intermediate evaluations).
//
Vec2 computeAcceleration(const std::vector<std::unique_ptr<Body>>& bodies, const std::vector<State>& states, int i) {
    Vec2 acc = {0, 0};          // Initialize acceleration to zero.
    const float softening = 10.0f;  // Softening factor avoids singularities.

    // Loop over every other body to sum their gravitational effect.
    for (int j = 0; j < states.size(); j++) {
        if (j == i) continue;   // Skip self.
        Vec2 d = states[j].pos - states[i].pos;  // Vector from body i to body j.
        float dist = length(d);  // Distance between body i and body j.
        float denom = (dist * dist) + (softening * softening);  // Adjusted denominator.
        float aMag = G * bodies[j]->mass / denom;  // Acceleration magnitude.
        // Add the acceleration contribution from body j (in the direction of the unit vector).
        acc = acc + normalize(d) * aMag;
    }
    return acc;
}


//
// rk4Step:
// This function performs a single RK4 integration step for all bodies.
// It computes the intermediate slopes (k1, k2, k3, k4) for both position and velocity,
// and then updates the state of each body using a weighted average of these slopes.
//
void rk4Step(std::vector<std::unique_ptr<Body>>& bodies, float dt) {
    int n = bodies.size();                 // Number of bodies.
    // Create vectors to hold the current state and the intermediate slopes.
    std::vector<State> s(n), k1(n), k2(n), k3(n), k4(n);

    // Initialize state vector s using the current positions and velocities of bodies.
    for (int i = 0; i < n; i++) {
        if (bodies[i]) {  // Check that the body exists.
            s[i].pos = bodies[i]->pos;
            s[i].vel = bodies[i]->vel;
        }
    }

    // --- Compute k1 = f(s)
    // Here, for each body, k1.pos is the current velocity,
    // and k1.vel is the computed acceleration at state s.
    for (int i = 0; i < n; i++) {
        if (bodies[i]) {
            k1[i].pos = s[i].vel;
            k1[i].vel = computeAcceleration(bodies, s, i);
        }
    }

    // Temporary state for RK4: s_temp = s + (k1 * dt/2)
    std::vector<State> s_temp(n);
    for (int i = 0; i < n; i++) {
        if (bodies[i]) {
            s_temp[i].pos = s[i].pos + k1[i].pos * (dt / 2.0f);
            s_temp[i].vel = s[i].vel + k1[i].vel * (dt / 2.0f);
        }
    }

    // --- Compute k2 = f(s + k1*dt/2)
    for (int i = 0; i < n; i++) {
        if (bodies[i]) {
            k2[i].pos = s_temp[i].vel;
            k2[i].vel = computeAcceleration(bodies, s_temp, i);
        }
    }

    // Recalculate s_temp = s + (k2 * dt/2) for k3 evaluation.
    for (int i = 0; i < n; i++) {
        if (bodies[i]) {
            s_temp[i].pos = s[i].pos + k2[i].pos * (dt / 2.0f);
            s_temp[i].vel = s[i].vel + k2[i].vel * (dt / 2.0f);
        }
    }

    // --- Compute k3 = f(s + k2*dt/2)
    for (int i = 0; i < n; i++) {
        if (bodies[i]) {
            k3[i].pos = s_temp[i].vel;
            k3[i].vel = computeAcceleration(bodies, s_temp, i);
        }
    }

    // Recalculate s_temp = s + (k3 * dt) for k4 evaluation.
    for (int i = 0; i < n; i++) {
        if (bodies[i]) {
            s_temp[i].pos = s[i].pos + k3[i].pos * dt;
            s_temp[i].vel = s[i].vel + k3[i].vel * dt;
        }
    }

    // --- Compute k4 = f(s + k3*dt)
    for (int i = 0; i < n; i++) {
        if (bodies[i]) {
            k4[i].pos = s_temp[i].vel;
            k4[i].vel = computeAcceleration(bodies, s_temp, i);
        }
    }

    // --- Combine the stages to update the state:
    // New state = s + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    for (int i = 0; i < n; i++) {
        if (bodies[i]) {
            State newState;
            newState.pos = s[i].pos + (k1[i].pos + k2[i].pos * 2.0f + k3[i].pos * 2.0f + k4[i].pos) * (dt / 6.0f);
            newState.vel = s[i].vel + (k1[i].vel + k2[i].vel * 2.0f + k3[i].vel * 2.0f + k4[i].vel) * (dt / 6.0f);
            bodies[i]->pos = newState.pos;   // Update the body's position.
            bodies[i]->vel = newState.vel;   // Update the body's velocity.
            bodies[i]->trail.push_back(newState.pos);  // Append new position to trail.
            if (bodies[i]->trail.size() > bodies[i]->maxTrailLength) {
                bodies[i]->trail.erase(bodies[i]->trail.begin()); // Maintain trail length.
            }
        }
    }
}


//------------------------------------------------------------------------------
// Main Simulation Function (Entry Point)
//------------------------------------------------------------------------------


Uint64 freq = SDL_GetPerformanceFrequency();
Uint64 now = SDL_GetPerformanceCounter();
Uint64 last = now;
float fps = 0;


int main(int argc, char* argv[]) {
    // Initialize SDL video subsystem and SDL_ttf.
    // Using the logical OR here ensures that if either initialization fails, we exit.
    if (SDL_Init(SDL_INIT_VIDEO) == 0 || TTF_Init() == 0) {
        std::cerr << "Initialization failed: " << SDL_GetError() << std::endl;
        return 1;
    }


    // Define window dimensions.
    const int winWidth = 1300;
    const int winHeight = 700;
    // Create a resizable window with the given width and height.
    SDL_Window* window = SDL_CreateWindow("Gravity Simulation", winWidth, winHeight, SDL_WINDOW_RESIZABLE);
    // Create a renderer for the window (using default parameters).
    SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr);
    // Open the font for UI text rendering.
    TTF_Font* font = TTF_OpenFont("OpenSans-Regular.ttf", 16);


    // Create stars in the background
    std::vector<SDL_Point> stars;
    for (int i = 0; i < 500; i++) {
        SDL_Point star;
        star.x = rand() % winWidth;
        star.y = rand() % winHeight;
        stars.push_back(star);
    }

    // Create a collection of celestial bodies. Typically a central massive "star" and orbiting bodies.
    std::vector<std::unique_ptr<Body>> bodies;
    bodies.push_back(std::make_unique<Body>(100000, Vec2{0, 0}, Vec2{0, 5}, SDL_Color{255, 0, 0, 255}, 50));
    bodies.push_back(std::make_unique<Body>(5000, Vec2{120, 0}, Vec2{260, 5}, SDL_Color{0, 255, 0, 255}, 8));
    bodies.push_back(std::make_unique<Body>(2000, Vec2{200, 0}, Vec2{200, 4}, SDL_Color{0, 100, 255, 255}, 7));
    bodies.push_back(std::make_unique<Body>(1000, Vec2{300, 0}, Vec2{150, 3}, SDL_Color{255, 200, 0, 255}, 6));


    // Set the simulation time step.
    const float dt = 0.01f;
    int simStep = 0;                 // Step counter.
    SDL_Color white = {255, 255, 255, 255};  // Color for UI text.
    int selectedBody = 0;            // Currently selected body (modifiable via key input).
    bool running = true;             // Main loop flag.
    bool dragging = false;
    Vec2 previousMouseWorldPos;
    Vec2 currentMouseWorldPos;
    bool justGrabbed = false;           // Flag to indicate panning via mouse dragging.
    float lastMouseX = 0.0f, lastMouseY = 0.0f;  // Last recorded mouse positions (for panning and planet stealing).


    Vec2 prevWorldPos = {0, 0};
    Vec2 lastWorldPos = {0, 0};


    // Lambda function to render UI text on the screen.
    // This function uses SDL_ttf to create a surface, then converts it to a texture,
    // and finally renders it on the screen at position (x, y).
    auto renderUIText = [&](const std::string& text, float x, float y) {
        // Note: Passing 0 for the background parameter means transparent background.
        SDL_Surface* surf = TTF_RenderText_Solid(font, text.c_str(), 0, white);
        if (!surf) return;
        SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
        SDL_FRect rect = {x, y, static_cast<float>(surf->w), static_cast<float>(surf->h)};
        SDL_RenderTexture(renderer, tex, nullptr, &rect);
        SDL_DestroyTexture(tex);
        SDL_DestroySurface(surf);
    };

    // Main simulation and rendering loop.
    bool draggingPlanet = true;
    int selectedPlanetIndex = -1;
    float grab_radius = 100.0f;
    Vec2 dragStartPos; // Where the drag began in world coordinates
    Vec2 dragPrevPos;  // Last frame's position while dragging



    while (running) {
        SDL_Event event;
        // Process all pending events.
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_EVENT_QUIT:  // Handle quit event (e.g., window close button).
                    running = false;
                    break;
                case SDL_EVENT_KEY_DOWN:  // Handle key press events.
                    switch (event.key.key) {  // Use event.key.keysym.sym to get the key identifier.
                        case SDLK_ESCAPE: running = false; break;  // Exit simulation.
                        case SDLK_R: resetSimulation(bodies); break; // Reset simulation
                        case SDLK_G: G += 0.5f; break;               // Increase gravitational constant.
                        case SDLK_H: if (G > 0.5f) G -= 0.5f; break;   // Decrease gravitational constant.
                        case SDLK_1: selectedBody = 0; break;          // Select first body.
                        case SDLK_2: if (bodies.size() > 1) selectedBody = 1; break;
                        case SDLK_3: if (bodies.size() > 2) selectedBody = 2; break;
                        case SDLK_4: if (bodies.size() > 3) selectedBody = 3; break;
                        case SDLK_UP: bodies[selectedBody]->mass += 500; break;    // Increase mass.
                        case SDLK_DOWN:
                            if (bodies[selectedBody]->mass > 500)
                                bodies[selectedBody]->mass -= 500;                // Decrease mass.
                            break;
                        case SDLK_Z: zoom *= 1.1f; break;             // Zoom in.
                        case SDLK_X: zoom /= 1.1f; break;             // Zoom out.
                    }
                    break;
                case SDL_EVENT_MOUSE_WHEEL: {  // Handle zooming with mouse wheel.
                    SDL_FPoint mouse;
                    SDL_GetMouseState(&mouse.x, &mouse.y);  // Get current mouse coordinates.
                    Vec2 beforeZoom = screenToWorld(mouse.x, mouse.y, winWidth, winHeight);
                    // Determine zoom factor based on wheel direction.
                    float zoomFactor = (event.wheel.y > 0) ? 1.1f : 0.9f;
                    zoom *= zoomFactor;  // Adjust zoom.
                    Vec2 afterZoom = screenToWorld(mouse.x, mouse.y, winWidth, winHeight);
                    // Adjust offset so that the world under the mouse remains fixed.
                    offset = offset + (beforeZoom - afterZoom);
                }
                break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:  // Handle mouse press events for dragging.
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        dragging = true;
                        lastMouseX = event.button.x;
                        lastMouseY = event.button.y;
                    }
                    if (event.button.button == SDL_BUTTON_RIGHT) {
                        SDL_FPoint mousePos;
                        SDL_GetMouseState(&mousePos.x, &mousePos.y);
                        Vec2 worldPos = screenToWorld(mousePos.x, mousePos.y, winWidth, winHeight);

                        // Check if the mouse is over any planet
                        for (size_t i = 0; i < bodies.size(); ++i) {
                            Body* body = bodies[i].get();
                            float dx = worldPos.x - body->pos.x;
                            float dy = worldPos.y - body->pos.y;
                            float grabRadius = body->radius + grab_radius;
                            if (sqrt(dx * dx + dy * dy) <= grabRadius) {
                                draggingPlanet = true;
                                selectedPlanetIndex = i;
                                lastMouseX = mousePos.x;
                                lastMouseY = mousePos.y;
                                dragStartPos = worldPos;
                                dragPrevPos = worldPos;
                                bodies[selectedPlanetIndex]->vel = Vec2{0, 0}; // Stop movement

                                break;
                            }
                        }
                    }

                    break;
                case SDL_EVENT_MOUSE_BUTTON_UP:    // Stop dragging when mouse button released.
                    if (event.button.button == SDL_BUTTON_LEFT) dragging = false;

                    if (event.button.button == SDL_BUTTON_RIGHT) {
                        if (draggingPlanet && selectedPlanetIndex != -1) {
                            SDL_FPoint mousePos;
                            SDL_GetMouseState(&mousePos.x, &mousePos.y);
                            Vec2 worldPos = screenToWorld(mousePos.x, mousePos.y, winWidth, winHeight);

                            // Calculate velocity based on drag difference
                            Vec2 dragVelocity = (worldPos - dragStartPos) * 1.1f; // Tune the multiplier if needed
                            bodies[selectedPlanetIndex]->vel = dragVelocity;
                        }
                        draggingPlanet = false;
                        selectedPlanetIndex = -1;

                    }

                    case SDL_EVENT_MOUSE_MOTION:       // Handle mouse movement for panning and dragging.
                    if (dragging) {
                        int dx = event.motion.x - lastMouseX;
                        int dy = event.motion.y - lastMouseY;
                        // Adjust offset by the mouse movement, factoring in the zoom level.
                        offset.x += dx / zoom;
                        offset.y -= dy / zoom;
                        lastMouseX = event.motion.x;
                        lastMouseY = event.motion.y;
                    }
                    if (draggingPlanet && selectedPlanetIndex != -1) {
                        SDL_FPoint mousePos;
                        SDL_GetMouseState(&mousePos.x, &mousePos.y);
                        Vec2 worldPos = screenToWorld(mousePos.x, mousePos.y, winWidth, winHeight);

                        Body* body = bodies[selectedPlanetIndex].get();
                        Vec2 toMouse = worldPos - body->pos;

                        // Softly drag towards the mouse
                        body->vel = toMouse * 3.0f; // Planet velocity towards mouse
                        body->pos = body->pos + toMouse * 0.2f; // Lerp factor
                    }
                    break;
            }
        }

        // Use RK4 integration to update all bodies for this time step dt.
        rk4Step(bodies, dt);
        simStep++;  // Increment simulation step counter.

        // Clear the screen with a background color.
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 150);
        SDL_RenderClear(renderer);


        // Draw stars
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        for (const auto& star : stars) {
            SDL_RenderPoint(renderer, star.x, star.y);
        }

        // Draw light rays (from the brightest star) onto the scene.
        drawLightRays(renderer, bodies, winWidth, winHeight);
        cullBodies(bodies,winWidth,winHeight);

        // For each body, render its trail, the body itself as a filled circle,
        // and its velocity vector.
        for (const auto& body : bodies) {
            //debug line under
            //std::cout<<"is body visible? :"+to_string(body->isVisible)<<std::endl;
            if (body->isVisible) { // Check if the body is
                drawTrail(renderer, body.get(), winWidth, winHeight);
                SDL_FPoint screenPos = toScreen(body->pos, winWidth, winHeight);
                drawFilledCircle(renderer, screenPos.x, screenPos.y, body->radius * zoom, body->color);
                drawVelocityVector(renderer, body->pos, body->vel, winWidth, winHeight);
            }
        }


        handleCollisions(bodies);

        last = now;
        now = SDL_GetPerformanceCounter();
        float elapsedMS = (now - last) * 1000.0f / freq;
        fps = 1000.0f / elapsedMS;


        // Render UI text information (simulation step, body masses, gravitational constant, and controls).
        renderUIText("Step: " + to_string(simStep), 10, 10);
        renderUIText("FPS: "+to_string(fps), 600, 10);
        for (size_t i = 0; i < bodies.size(); ++i) {
            renderUIText("Body " + to_string((int)i + 1) + " with mass: " + to_string((int)bodies[i]->mass)+" and position: "+to_string((int)bodies[i]->pos.x)+" , "+to_string((int)bodies[i]->pos.y) ,
                10, 30 + 15 * i);
        }
        renderUIText("G: " + to_string(G), 10, 30 + 15 * bodies.size());
        renderUIText("Use mouse wheel to zoom, drag to pan.", 10, winHeight - 90);
        renderUIText("Controls: UP/DOWN to change masses | G/H to change gravitational constant", 10, winHeight - 70);
        renderUIText("Change body by pressing 1/2/3/4", 10, winHeight - 50);
        renderUIText("Selected body is: " + to_string(selectedBody + 1), 10, winHeight - 30);

        // Present the rendered frame on the display.
        SDL_RenderPresent(renderer);
        SDL_Delay(16); // Delay ~16ms to target approximately 60 FPS.
    }

    // Cleanup: free memory for all bodies and SDL resources.
    bodies.clear();
    TTF_CloseFont(font);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();
    return 0;
}
