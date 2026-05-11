const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('path');
const fs   = require('fs');

const DEFAULT_SERVER_URL = 'http://badmaevoleg.hlab.kz';
const CONFIG_FILE = path.join(app.getPath('userData'), 'config.json');

let mainWindow = null;

function readConfig() {
  try { return JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8')); }
  catch (_) { return {}; }
}

function writeConfig(cfg) {
  fs.mkdirSync(path.dirname(CONFIG_FILE), { recursive: true });
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(cfg, null, 2), 'utf8');
}

function createMain(url) {
  mainWindow = new BrowserWindow({
    width: 1440, height: 900,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });
  mainWindow.loadURL(url);
  mainWindow.on('closed', () => { mainWindow = null; });
  mainWindow.webContents.setWindowOpenHandler(({ url: u }) => {
    shell.openExternal(u); return { action: 'deny' };
  });
}

app.whenReady().then(() => {
  const cfg = readConfig();
  const url = cfg.serverUrl || DEFAULT_SERVER_URL;
  if (!cfg.serverUrl) { cfg.serverUrl = url; writeConfig(cfg); }
  createMain(url);
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });

ipcMain.handle('get-version',    () => app.getVersion());
ipcMain.handle('get-server-url', () => readConfig().serverUrl || DEFAULT_SERVER_URL);
ipcMain.on('set-server-url', (_, url) => {
  let u = url.trim().replace(/\/$/, '');
  if (u && !u.startsWith('http://') && !u.startsWith('https://')) u = 'http://' + u;
  const cfg = readConfig(); cfg.serverUrl = u; writeConfig(cfg);
  if (mainWindow) mainWindow.loadURL(u);
});
ipcMain.handle('reset-server-url', () => {
  const cfg = readConfig(); delete cfg.serverUrl; writeConfig(cfg);
  if (mainWindow) mainWindow.loadURL(DEFAULT_SERVER_URL);
});
