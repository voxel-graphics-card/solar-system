# 🌌 Gravity Simulation in C++ with SDL #

A real-time 2D gravity simulation built in C++ using SDL3. This project simulates celestial bodies influenced by gravitational forces, with support for interaction, collision, and beautiful orbital trails.

 ![nice](https://github.com/user-attachments/assets/35b85514-7140-4373-9488-cd059e38127d)


---

##  Features

-  Newtonian gravity simulation between bodies  
-  Realistic gravitational collisions & merging  
-  Planet repulsion if both are small bodies  
-  Click & drag planets to reposition  
-  Fling planets with the mouse to set new trajectories  
-  Smooth panning & zooming support  
-  Velocity vectors and trails for each body  
-  Reset simulation and tweak gravity on the fly

---

## How It Works

- Uses **Runge-Kutta 4th Order Integration (RK4)** for accurate physics
- All rendering and input handled through **SDL3**
- Bodies are stored as `std::unique_ptr<Body>` to manage memory safely
- Zoom and offset logic for intuitive world navigation

---

## 🕹 Controls

| Action                | Key / Mouse                  |
|-----------------------|------------------------------|
| Pan view              | Drag with left mouse button  |
| Zoom in/out           | Scroll wheel or `Z` / `X`    |
| Drag planet           | Right-click and move mouse   |
| Fling planet          | Release right-click after dragging |
| Reset simulation      | `R`                          |
| Select body           | `1` `2` `3` `4`              |
| Increase/Decrease mass| Up / Down arrows             |
| Increase/Decrease G   | `G` / `H`                    |
| Exit                  | `Esc`                        |

---

## Requirements

- C++17 or later
- [SDL3](https://github.com/libsdl-org/SDL)
- [SDL_ttf](https://github.com/libsdl-org/SDL_ttf)

---

##  Build Instructions on windows with MINGW

g++ -std=c++17 -o gravity_sim *.cpp -lSDL3 -lSDL3_ttf
./gravity_sim
