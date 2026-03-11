const { spawn } = require("child_process");

// Some shells export this and force Electron to run as plain Node.
delete process.env.ELECTRON_RUN_AS_NODE;

const electronPath = require("electron");
const child = spawn(electronPath, ["."], {
  stdio: "inherit",
  env: process.env,
});

child.on("exit", (code) => {
  process.exit(code ?? 0);
});

