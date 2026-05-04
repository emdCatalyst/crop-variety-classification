/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: { center: true, padding: "1rem" },
    extend: {
      colors: {
        brand: {
          50: "#f3f9f1",
          100: "#dcefd5",
          500: "#3f8a3a",
          600: "#2f6c2c",
          700: "#235021",
        },
      },
    },
  },
  plugins: [require("tailwindcss-rtl")],
};
