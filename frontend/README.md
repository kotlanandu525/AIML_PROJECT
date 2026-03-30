<div align="center">
  <h2>💜 Heart Stroke Dashboard Frontend</h2>
  <p><strong>A beautifully crafted, modern user interface for exploring cardiovascular risk models natively via React and Vite.</strong></p>
</div>

## 📌 Context
This subdirectory represents the presentation layer (Client UI) of the Heart Stroke Prediction AI project. It was meticulously generated using Vite to facilitate immediate Hot Module Replacement, robust builds, and a lightweight development configuration.

## 🎨 Aesthetics
The focus of this interface heavily emphasizes **Visual Excellence** showcasing elements of:
- **Glassmorphism**: Using `backdrop-filter` utility principles on semitransparent layers for a frosted glass look.
- **Deep Gradient Styling**: Smooth, complementary gradients and organic shapes as aesthetic backdrops.
- **Fluid Micro-Animations**: Interactive hover, transition, and scale responses that make the software feel natively alive.
- **Dynamic Charts**: Powered by `recharts`, ensuring numerical data from the machine learning APIs are visualized simply and comparatively against average population sizes.

## ⚙️ Core Stack
- **Library**: `React`
- **Builder**: `Vite` 
- **Graphing**: `Recharts`
- **Styling**: `Vanilla CSS`
- **Networking**: Native Javascript `Fetch API` for consuming FastAPI prediction instances from the backend server.

## 🔧 Installation & Usage
The local dependencies for this React frontend can be installed via your favorite node package manager.

### Install Node modules
```bash
npm install
```

### Start Local Hot-Reload Server
```bash
npm run dev
```

### Compile Production Build
If you intend to host the static Javascript and CSS chunks within Nginx or an alternative static proxy server:
```bash
npm run build
```

The resulting `dist` folder will compile a fully minimized, high-performance static rendering of the application UI layer. No environment variable injection is strictly required unless switching API remote origins.

> For holistic environment startup (combining backend ML servers), please reference the central parent `README.md`.
