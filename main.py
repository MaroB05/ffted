import sys
import numpy as np
import cv2
from scipy.fft import dct, idct
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QRadioButton, QButtonGroup,
                             QFileDialog, QLabel, QMessageBox, QCheckBox, QScrollArea,
                             QLineEdit, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import threading
import time


class ComputeWorker(QObject):
    result_ready = pyqtSignal(object)  # emits a dict with results
    dct_result_ready = pyqtSignal(object)  # emits a dict with DCT results
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._dct_cancel_event = threading.Event()
        self._pending = None  # latest-only request dict
        self._pending_dct = None  # latest-only DCT request dict
        self._stop = False
    
    def submit_latest(self, request_dict):
        with self._lock:
            self._pending = request_dict
        self._cancel_event.set()  # cancel any in-flight work
    
    def submit_latest_dct(self, request_dict):
        with self._lock:
            self._pending_dct = request_dict
        self._dct_cancel_event.set()  # cancel any in-flight DCT work
    
    def stop(self):
        self._stop = True
        self._cancel_event.set()
        self._dct_cancel_event.set()
    
    def _should_cancel(self, req_id: int) -> bool:
        if self._cancel_event.is_set():
            return True
        with self._lock:
            return self._pending is not None and self._pending['req_id'] > req_id
    
    def _should_cancel_dct(self, req_id: int) -> bool:
        if self._dct_cancel_event.is_set():
            return True
        with self._lock:
            return self._pending_dct is not None and self._pending_dct['req_id'] > req_id
    
    def run(self):
        while not self._stop:
            # Process FFT request
            with self._lock:
                req = self._pending
                self._pending = None
            if req is not None:
                self._cancel_event.clear()
                try:
                    req_id = req['req_id']
                    image = req['image']
                    fft_shift = req['fft_shift']
                    typed_filters = req['typed_filters']  # list of (kind, low, high)
                    sample_step = max(1, req['sample_step'])
                    log_scale = bool(req.get('log_scale', True))
                    viz_mode = req.get('viz_mode', 'unit')  # 'unit' | 'mag' | 'normalized'
                    
                    # Stage 1: FFT (+ optional shift)
                    if self._should_cancel(req_id): 
                        continue
                    fft = np.fft.fft2(image)
                    base_fft = np.fft.fftshift(fft) if fft_shift else fft
                    
                    # Stage 2: frequency grids
                    if self._should_cancel(req_id): 
                        continue
                    h, w = image.shape
                    fy = np.fft.fftfreq(h)
                    fx = np.fft.fftfreq(w)
                    FX, FY = np.meshgrid(fx, fy)
                    radius = np.sqrt(FX**2 + FY**2)
                    if fft_shift:
                        radius = np.fft.fftshift(radius)
                    
                    # Stage 3: filtering
                    if self._should_cancel(req_id): 
                        continue
                    mag = np.abs(base_fft)
                    mask = np.ones_like(base_fft, dtype=np.float32)
                    for kind, low, high in typed_filters:
                        if kind == 'freq':
                            band = (radius >= low) & (radius <= high)
                        else:
                            band = (mag >= low) & (mag <= high)
                        mask[band] = 0.0
                    filtered_fft = base_fft * mask
                    
                    # Stage 4: reconstruction
                    if self._should_cancel(req_id): 
                        continue
                    spectrum = np.fft.ifftshift(filtered_fft) if fft_shift else filtered_fft
                    recon = np.real(np.fft.ifft2(spectrum))
                    rmin, rmax = recon.min(), recon.max()
                    recon_norm = (recon - rmin) / (rmax - rmin) if rmax > rmin else np.zeros_like(recon)
                    recon_img = (recon_norm * 255).astype(np.uint8)
                    
                    # Stage 5: PSD + log
                    if self._should_cancel(req_id): 
                        continue
                    psd = np.abs(filtered_fft) ** 2
                    psd_log = np.log10(psd + 1)
                    psd_display = psd_log if log_scale else psd
                    psd_label = 'Log PSD' if log_scale else 'PSD'
                    
                    # Stage 6: sampled points for 3D (optional, UI may resample itself)
                    if self._should_cancel(req_id): 
                        continue
                    f_sampled = filtered_fft[::sample_step, ::sample_step]
                    real = np.real(f_sampled).flatten()
                    imag = np.imag(f_sampled).flatten()
                    mag_sampled = np.abs(f_sampled).flatten()
                    # Compute per-UI options: log magnitude for height/colors
                    mag_for_height = np.log10(mag_sampled + 1) if log_scale else mag_sampled
                    # Mode-specific transforms
                    if viz_mode == 'unit':
                        z_vals = np.ones_like(mag_for_height)
                        rx, ry = real, imag
                        title = '3D FFT Visualization (Unit Height)'
                    elif viz_mode == 'mag':
                        z_vals = mag_for_height
                        rx, ry = real, imag
                        title = '3D FFT Visualization (Magnitude Height)'
                    else:
                        norm = np.sqrt(real**2 + imag**2)
                        norm = np.where(norm == 0, 1, norm)
                        rx = real / norm
                        ry = imag / norm
                        z_vals = mag_for_height
                        title = '3D FFT Visualization (Normalized + Magnitude)'
                    # Colors follow log setting
                    colors = mag_for_height if log_scale else mag_sampled
                    # Axis limits based on percentiles (for main-thread-only set)
                    real_min, real_max = np.percentile(rx, [1, 99]) if rx.size else (0, 1)
                    imag_min, imag_max = np.percentile(ry, [1, 99]) if ry.size else (0, 1)
                    z_min, z_max = np.percentile(z_vals, [1, 99]) if z_vals.size else (0, 1)
                    limits = {
                        'real_min': float(real_min), 'real_max': float(real_max),
                        'imag_min': float(imag_min), 'imag_max': float(imag_max),
                        'z_min': float(z_min), 'z_max': float(z_max),
                        'title': title,
                        'colorbar_label': ('Log Magnitude' if log_scale else 'Magnitude'),
                    }
                    
                    self.result_ready.emit({
                        'req_id': req_id,
                        'base_fft': base_fft,
                        'filtered_fft': filtered_fft,
                        'recon_img': recon_img,
                        'psd': psd,
                        'psd_log': psd_log,
                        'psd_display': psd_display,
                        'psd_label': psd_label,
                        'sampled': {
                            'real': rx, 'imag': ry, 'mag': mag_sampled,
                            'z': z_vals, 'colors': colors, 'limits': limits
                        },
                    })
                except Exception as e:
                    self.error.emit(str(e))
            
            # Process DCT request
            with self._lock:
                req_dct = self._pending_dct
                self._pending_dct = None
            if req_dct is not None:
                self._dct_cancel_event.clear()
                try:
                    req_id = req_dct['req_id']
                    image = req_dct['image']
                    sample_step = max(1, req_dct['sample_step'])
                    log_scale = bool(req_dct.get('log_scale', True))
                    viz_mode = req_dct.get('viz_mode', 'unit')
                    
                    # Compute 2D DCT
                    if self._should_cancel_dct(req_id):
                        continue
                    dct_result = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
                    
                    # Reconstruct image from DCT
                    if self._should_cancel_dct(req_id):
                        continue
                    recon_dct = idct(idct(dct_result, axis=0, norm='ortho'), axis=1, norm='ortho')
                    # Normalize for display to 0-255
                    rmin, rmax = recon_dct.min(), recon_dct.max()
                    recon_norm = (recon_dct - rmin) / (rmax - rmin) if rmax > rmin else np.zeros_like(recon_dct)
                    recon_dct_img = (recon_norm * 255).astype(np.uint8)
                    
                    # Sample for 3D visualization
                    if self._should_cancel_dct(req_id):
                        continue
                    dct_sampled = dct_result[::sample_step, ::sample_step]
                    # DCT is real-valued, so we use the DCT coefficients directly
                    # For 3D visualization, we can use: x = frequency index, y = frequency index, z = magnitude
                    h_s, w_s = dct_sampled.shape
                    x_coords = np.tile(np.arange(w_s), h_s)  # frequency index X
                    y_coords = np.repeat(np.arange(h_s), w_s)  # frequency index Y
                    mag_dct = np.abs(dct_sampled).flatten()
                    
                    # Apply log scale if needed
                    mag_for_height = np.log10(mag_dct + 1) if log_scale else mag_dct
                    
                    # Mode-specific transforms
                    if viz_mode == 'unit':
                        z_vals = np.ones_like(mag_for_height)
                        rx, ry = x_coords, y_coords
                        title = '3D DCT Visualization (Unit Height)'
                    elif viz_mode == 'mag':
                        z_vals = mag_for_height
                        rx, ry = x_coords, y_coords
                        title = '3D DCT Visualization (Magnitude Height)'
                    else:
                        # For normalized mode, normalize x,y to unit circle
                        norm = np.sqrt(x_coords**2 + y_coords**2)
                        norm = np.where(norm == 0, 1, norm)
                        rx = x_coords / norm
                        ry = y_coords / norm
                        z_vals = mag_for_height
                        title = '3D DCT Visualization (Normalized + Magnitude)'
                    
                    # Colors follow log setting
                    colors = mag_for_height if log_scale else mag_dct
                    
                    # Axis limits
                    x_min, x_max = np.percentile(rx, [1, 99]) if rx.size else (0, 1)
                    y_min, y_max = np.percentile(ry, [1, 99]) if ry.size else (0, 1)
                    z_min, z_max = np.percentile(z_vals, [1, 99]) if z_vals.size else (0, 1)
                    limits = {
                        'real_min': float(x_min), 'real_max': float(x_max),
                        'imag_min': float(y_min), 'imag_max': float(y_max),
                        'z_min': float(z_min), 'z_max': float(z_max),
                        'title': title,
                        'colorbar_label': ('Log Magnitude' if log_scale else 'Magnitude'),
                    }
                    
                    self.dct_result_ready.emit({
                        'req_id': req_id,
                        'dct_data': dct_result,
                        'recon_dct_img': recon_dct_img,
                        'sampled': {
                            'real': rx, 'imag': ry, 'mag': mag_dct,
                            'z': z_vals, 'colors': colors, 'limits': limits
                        },
                    })
                except Exception as e:
                    self.error.emit(str(e))
            
            if req is None and req_dct is None:
                time.sleep(0.01)
 
class FFTVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.image_original = None
        self.original_shape = None
        self.reconstructed_image = None
        self.base_fft_data = None
        self.filtered_fft_data = None
        self.frequency_radius = None
        self.filter_rows = []
        # Cached arrays to avoid redundant heavy recomputation
        self._cached_psd = None           # |FFT|^2 for current filtered spectrum
        self._cached_psd_log = None       # log10(|FFT|^2 + 1)
        # Cached artists for faster redraws
        self._plot_ax = None
        self._scatter_artist = None
        self._plot_colorbar = None
        self._psd_ax = None
        self._psd_im = None
        self._psd_colorbar = None
        # Sampled points provided by worker for 3D plotting
        self._sampled_real = None
        self._sampled_imag = None
        self._sampled_mag = None
        # DCT state
        self.dct_data = None
        self.dct_reconstructed_image = None
        self._dct_plot_ax = None
        self._dct_scatter_artist = None
        self._dct_plot_colorbar = None
        self._dct_sampled_real = None
        self._dct_sampled_imag = None
        self._dct_sampled_mag = None
        self._dct_sampled_z = None
        self._dct_sampled_colors = None
        self._dct_sampled_limits = None
        self._next_dct_req_id = 1
        self._latest_dct_req_id = 0
        self.initUI()
        
        # Background compute thread setup
        self._thread = QThread(self)
        self._worker = ComputeWorker()
        self._worker.moveToThread(self._thread)
        self._worker.result_ready.connect(self._on_compute_result)
        self._worker.dct_result_ready.connect(self._on_dct_compute_result)
        self._worker.error.connect(lambda msg: self.status_label.setText(f'Error: {msg}'))
        self._thread.started.connect(self._worker.run)
        self._thread.start()
        self._next_req_id = 1
        self._latest_req_id = 0
        
    def initUI(self):
        self.setWindowTitle('Image FFT 3D Visualizer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout (horizontal)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Content area (left side) - contains image and 3D plot
        content_layout = QVBoxLayout()
        
        # 3D plots side-by-side (FFT left, DCT right)
        plots_widget = QWidget()
        plots_layout = QHBoxLayout()
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(8)
        plots_widget.setLayout(plots_layout)
        
        # FFT 3D plot canvas (left half)
        self.plot_figure = Figure(figsize=(4, 5))
        self.plot_canvas = FigureCanvas(self.plot_figure)
        plots_layout.addWidget(self.plot_canvas, stretch=1)
        
        # DCT 3D plot canvas (right half)
        self.dct_plot_figure = Figure(figsize=(4, 5))
        self.dct_plot_canvas = FigureCanvas(self.dct_plot_figure)
        plots_layout.addWidget(self.dct_plot_canvas, stretch=1)
        
        content_layout.addWidget(plots_widget, stretch=3)
        
        # PSD plot canvas (middle)
        self.psd_figure = Figure(figsize=(8, 3))
        self.psd_canvas = FigureCanvas(self.psd_figure)
        content_layout.addWidget(self.psd_canvas, stretch=2)
        
        # Original and reconstructed image canvases (bottom side-by-side: Original, FFT Recon, DCT Recon)
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)
        bottom_widget.setLayout(bottom_layout)
        
        self.image_figure = Figure(figsize=(3, 3))
        self.image_canvas = FigureCanvas(self.image_figure)
        bottom_layout.addWidget(self.image_canvas, stretch=1)
        
        self.recon_figure = Figure(figsize=(3, 3))
        self.recon_canvas = FigureCanvas(self.recon_figure)
        bottom_layout.addWidget(self.recon_canvas, stretch=1)
        
        self.dct_recon_figure = Figure(figsize=(3, 3))
        self.dct_recon_canvas = FigureCanvas(self.dct_recon_figure)
        bottom_layout.addWidget(self.dct_recon_canvas, stretch=1)
        
        content_layout.addWidget(bottom_widget, stretch=2)
        
        # Status label at bottom
        self.status_label = QLabel('Load an image to begin')
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.status_label)
        
        main_layout.addLayout(content_layout, stretch=4)
        
        # Control panel (right side vertical bar)
        control_widget = QWidget()
        control_widget.setMaximumWidth(280)
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)
        
        # Load image button
        self.load_btn = QPushButton('Load Image')
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)
        
        # Compute FFT button
        self.compute_btn = QPushButton('Compute FFT')
        self.compute_btn.clicked.connect(self.compute_fft)
        self.compute_btn.setEnabled(False)
        control_layout.addWidget(self.compute_btn)
        
        # Compute DCT button
        self.compute_dct_btn = QPushButton('Compute DCT')
        self.compute_dct_btn.clicked.connect(self.compute_dct)
        self.compute_dct_btn.setEnabled(False)
        control_layout.addWidget(self.compute_dct_btn)
        
        # Export image button
        self.export_btn = QPushButton('Export Image')
        self.export_btn.clicked.connect(self.export_image)
        self.export_btn.setEnabled(False)
        control_layout.addWidget(self.export_btn)
        
        control_layout.addSpacing(20)
        
        # Separator label
        mode_label = QLabel('Visualization Mode:')
        mode_label.setStyleSheet('font-weight: bold;')
        control_layout.addWidget(mode_label)
        
        # Radio buttons for visualization mode
        self.radio_group = QButtonGroup()
        
        self.radio_unit_height = QRadioButton('Unit Height (z=1)')
        self.radio_unit_height.setChecked(True)
        self.radio_unit_height.toggled.connect(self._on_viz_option_changed)
        self.radio_group.addButton(self.radio_unit_height)
        control_layout.addWidget(self.radio_unit_height)
        
        self.radio_magnitude_height = QRadioButton('Magnitude Height')
        self.radio_magnitude_height.toggled.connect(self._on_viz_option_changed)
        self.radio_group.addButton(self.radio_magnitude_height)
        control_layout.addWidget(self.radio_magnitude_height)
        
        self.radio_normalized = QRadioButton('Normalized + Magnitude')
        self.radio_normalized.toggled.connect(self._on_viz_option_changed)
        self.radio_group.addButton(self.radio_normalized)
        control_layout.addWidget(self.radio_normalized)
        
        control_layout.addSpacing(20)
        
        # Options label
        options_label = QLabel('Options:')
        options_label.setStyleSheet('font-weight: bold;')
        control_layout.addWidget(options_label)
        
        # Log scale checkbox
        self.log_scale_checkbox = QCheckBox('Log Scale Magnitudes')
        self.log_scale_checkbox.setChecked(True)
        self.log_scale_checkbox.toggled.connect(self._on_viz_option_changed)
        control_layout.addWidget(self.log_scale_checkbox)
        
        # FFT shift checkbox
        self.fft_shift_checkbox = QCheckBox('Apply FFT Shift')
        self.fft_shift_checkbox.setChecked(True)
        self.fft_shift_checkbox.toggled.connect(self.compute_fft)
        control_layout.addWidget(self.fft_shift_checkbox)
        
        # Resize checkbox
        self.resize_checkbox = QCheckBox('Resize to 128x128')
        self.resize_checkbox.setChecked(True)
        self.resize_checkbox.toggled.connect(self.on_resize_toggle)
        control_layout.addWidget(self.resize_checkbox)
        
        control_layout.addSpacing(12)
        
        # Filters section
        filters_label = QLabel('Frequency Filters:')
        filters_label.setStyleSheet('font-weight: bold;')
        control_layout.addWidget(filters_label)
        
        # Add filter button
        add_filter_btn = QPushButton('+ Add filter')
        add_filter_btn.clicked.connect(self.add_filter_row)
        control_layout.addWidget(add_filter_btn)
        
        # Scrollable area for filters
        self.filters_scroll = QScrollArea()
        self.filters_scroll.setWidgetResizable(True)
        self.filters_scroll.setMinimumHeight(200)
        self.filters_scroll.setMinimumWidth(240)
        self.filters_container = QWidget()
        self.filters_layout = QVBoxLayout()
        self.filters_layout.setSpacing(6)
        self.filters_layout.setContentsMargins(0, 0, 0, 0)
        self.filters_container.setLayout(self.filters_layout)
        self.filters_scroll.setWidget(self.filters_container)
        control_layout.addWidget(self.filters_scroll, stretch=1)
        
        # Add an initial empty stretch to keep items compact at top
        self.filters_layout.addStretch()
        
        control_layout.addStretch()
        
        main_layout.addWidget(control_widget, stretch=1)
        
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)')
        
        if file_path:
            # Load image in grayscale
            loaded = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if loaded is None:
                QMessageBox.warning(self, 'Error', 'Could not load image')
                return
            
            self.original_shape = loaded.shape
            self.image_original = loaded
            self.reconstructed_image = None
            
            # Set processing image based on resize checkbox
            self.apply_resize_if_needed()
            ph, pw = self.image.shape
            self.status_label.setText(f'Image loaded: original {self.original_shape[1]}x{self.original_shape[0]}, processing {pw}x{ph}')
            
            # Display image
            self.image_figure.clear()
            ax = self.image_figure.add_subplot(111)
            ax.imshow(self.image, cmap='gray')
            ax.set_title('Original Image')
            ax.axis('off')
            self.image_canvas.draw()
            
            # Clear reconstructed images
            self.recon_figure.clear()
            axr = self.recon_figure.add_subplot(111)
            axr.set_title('FFT Reconstructed Image')
            axr.axis('off')
            self.recon_canvas.draw()
            
            self.dct_recon_figure.clear()
            axd = self.dct_recon_figure.add_subplot(111)
            axd.set_title('DCT Reconstructed Image')
            axd.axis('off')
            self.dct_recon_canvas.draw()
            
            # Enable compute buttons
            self.compute_btn.setEnabled(True)
            self.compute_dct_btn.setEnabled(True)
            self.export_btn.setEnabled(False)
            self.base_fft_data = None
            self.filtered_fft_data = None
            self.frequency_radius = None
            # Clear DCT data
            self.dct_data = None
            self.dct_reconstructed_image = None
            self._dct_sampled_real = None
            self._dct_sampled_imag = None
            self._dct_sampled_mag = None
            self._dct_sampled_z = None
            self._dct_sampled_colors = None
            self._dct_sampled_limits = None
            # Invalidate caches
            self._cached_psd = None
            self._cached_psd_log = None
            
            # Clear 3D plots
            self.plot_figure.clear()
            self.plot_canvas.draw()
            self.dct_plot_figure.clear()
            self.dct_plot_canvas.draw()
            
            # Clear PSD plot
            self.psd_figure.clear()
            self.psd_canvas.draw()
    
    def compute_fft(self):
        """Enqueue a compute request to the background worker."""
        self._enqueue_compute()
    
    def compute_dct(self):
        """Enqueue a DCT compute request to the background worker."""
        self._enqueue_dct_compute()
    
    def _on_viz_option_changed(self):
        """Handle visualization option changes (radio buttons, log scale)."""
        # Recompute FFT if FFT data exists
        if self.base_fft_data is not None:
            self._enqueue_compute()
        # Recompute DCT if DCT data exists
        if self.dct_data is not None:
            self._enqueue_dct_compute()
    
    def _build_typed_filters(self):
        typed = []
        for row in self.filter_rows:
            if not row['enable'].isChecked():
                continue
            low_text = row['low'].text().strip()
            high_text = row['high'].text().strip()
            if not low_text or not high_text:
                continue
            try:
                low_val = float(low_text)
                high_val = float(high_text)
            except ValueError:
                continue
            if high_val < low_val:
                low_val, high_val = high_val, low_val
            low_val = max(0.0, low_val)
            high_val = max(0.0, high_val)
            typed.append((row['type'].currentText(), low_val, high_val))
        return typed
    
    def _enqueue_compute(self):
        if self.image is None:
            return
        # Prepare request
        req_id = self._next_req_id
        self._next_req_id += 1
        self._latest_req_id = req_id
        
        h, w = self.image.shape
        sample_step = max(1, max(h, w) // 50)
        # Determine viz mode from radios
        if self.radio_unit_height.isChecked():
            viz_mode = 'unit'
        elif self.radio_magnitude_height.isChecked():
            viz_mode = 'mag'
        else:
            viz_mode = 'normalized'
        request = {
            'req_id': req_id,
            'image': self.image.copy(),
            'fft_shift': self.fft_shift_checkbox.isChecked(),
            'typed_filters': self._build_typed_filters(),
            'sample_step': sample_step,
            'log_scale': self.log_scale_checkbox.isChecked(),
            'viz_mode': viz_mode,
        }
        self.status_label.setText('Computing...')
        self._worker.submit_latest(request)
    
    def _enqueue_dct_compute(self):
        if self.image is None:
            return
        # Prepare DCT request
        req_id = self._next_dct_req_id
        self._next_dct_req_id += 1
        self._latest_dct_req_id = req_id
        
        h, w = self.image.shape
        sample_step = max(1, max(h, w) // 50)
        # Determine viz mode from radios
        if self.radio_unit_height.isChecked():
            viz_mode = 'unit'
        elif self.radio_magnitude_height.isChecked():
            viz_mode = 'mag'
        else:
            viz_mode = 'normalized'
        request = {
            'req_id': req_id,
            'image': self.image.copy(),
            'sample_step': sample_step,
            'log_scale': self.log_scale_checkbox.isChecked(),
            'viz_mode': viz_mode,
        }
        self.status_label.setText('Computing DCT...')
        self._worker.submit_latest_dct(request)
    
    def _on_compute_result(self, res: dict):
        # Drop stale result if a newer request exists
        if res.get('req_id') != self._latest_req_id:
            return
        # Update state from worker outputs
        self.base_fft_data = res['base_fft']
        self.filtered_fft_data = res['filtered_fft']
        self._cached_psd = res['psd']
        self._cached_psd_log = res['psd_log']
        self.reconstructed_image = res['recon_img']
        # Store sampled points for faster 3D updates
        self._sampled_real = res['sampled']['real']
        self._sampled_imag = res['sampled']['imag']
        self._sampled_mag = res['sampled']['mag']
        self._sampled_z = res['sampled']['z']
        self._sampled_colors = res['sampled']['colors']
        self._sampled_limits = res['sampled']['limits']
        self._psd_display = res['psd_display']
        self._psd_label_cached = res['psd_label']
        self.export_btn.setEnabled(True)
        
        # Render reconstructed image without recomputation
        self.recon_figure.clear()
        ax = self.recon_figure.add_subplot(111)
        ax.imshow(self.reconstructed_image, cmap='gray')
        ax.set_title('FFT Reconstructed Image')
        ax.axis('off')
        self.recon_canvas.draw()
        
        # Render 3D and PSD using current state
        self.update_plot()
    
    def update_plot(self):
        if self.filtered_fft_data is None:
            return
        
        self.status_label.setText('Rendering plots...')
        QApplication.processEvents()
        
        # Use sampled points and precomputed transforms from worker
        real = getattr(self, '_sampled_real', None)
        imag = getattr(self, '_sampled_imag', None)
        z = getattr(self, '_sampled_z', None)
        colors = getattr(self, '_sampled_colors', None)
        limits = getattr(self, '_sampled_limits', None)
        if real is None or imag is None or z is None or colors is None or limits is None:
            return
        
        title = limits.get('title', '')
        if self._plot_ax is None:
            self._plot_ax = self.plot_figure.add_subplot(111, projection='3d')
            self._scatter_artist = self._plot_ax.scatter(real, imag, z, c=colors, cmap='viridis',
                                                         marker='.', s=1, alpha=0.6)
            self._plot_ax.set_xlabel('Real')
            self._plot_ax.set_ylabel('Imaginary')
            self._plot_ax.set_zlabel('Height')
            self._plot_ax.set_title(title)
            # Set limits from worker-provided percentiles
            real_min = limits['real_min']; real_max = limits['real_max']
            imag_min = limits['imag_min']; imag_max = limits['imag_max']
            z_min = limits['z_min']; z_max = limits['z_max']
            padding = 0.1
            self._plot_ax.set_xlim(real_min - padding * (real_max - real_min), real_max + padding * (real_max - real_min))
            self._plot_ax.set_ylim(imag_min - padding * (imag_max - imag_min), imag_max + padding * (imag_max - imag_min))
            self._plot_ax.set_zlim(z_min - padding * (z_max - z_min), z_max + padding * (z_max - z_min))
            self._plot_colorbar = self.plot_figure.colorbar(self._scatter_artist, ax=self._plot_ax,
                                                            label=limits.get('colorbar_label', 'Magnitude'),
                                                            shrink=0.5)
        else:
            # Update data without recreating axes
            # Update 3D positions
            self._scatter_artist._offsets3d = (real, imag, z)
            # Update colors
            self._scatter_artist.set_array(colors)
            self._plot_ax.set_title(title)
            # Adjust limits from worker-provided bounds
            real_min = limits['real_min']; real_max = limits['real_max']
            imag_min = limits['imag_min']; imag_max = limits['imag_max']
            z_min = limits['z_min']; z_max = limits['z_max']
            padding = 0.1
            self._plot_ax.set_xlim(real_min - padding * (real_max - real_min), real_max + padding * (real_max - real_min))
            self._plot_ax.set_ylim(imag_min - padding * (imag_max - imag_min), imag_max + padding * (imag_max - imag_min))
            self._plot_ax.set_zlim(z_min - padding * (z_max - z_min), z_max + padding * (z_max - z_min))
            # Update colorbar mapping
            if self._plot_colorbar is not None:
                self._plot_colorbar.update_normal(self._scatter_artist)
        
        self.plot_canvas.draw()
        
        # Compute and plot PSD
        self.plot_psd()
        
        self.status_label.setText('Visualization complete')
    
    def _on_dct_compute_result(self, res: dict):
        # Drop stale result if a newer request exists
        if res.get('req_id') != self._latest_dct_req_id:
            return
        # Update DCT state from worker outputs
        self.dct_data = res['dct_data']
        self.dct_reconstructed_image = res['recon_dct_img']
        # Store sampled points for faster 3D updates
        self._dct_sampled_real = res['sampled']['real']
        self._dct_sampled_imag = res['sampled']['imag']
        self._dct_sampled_mag = res['sampled']['mag']
        self._dct_sampled_z = res['sampled']['z']
        self._dct_sampled_colors = res['sampled']['colors']
        self._dct_sampled_limits = res['sampled']['limits']
        
        # Render DCT reconstructed image
        self.dct_recon_figure.clear()
        ax = self.dct_recon_figure.add_subplot(111)
        ax.imshow(self.dct_reconstructed_image, cmap='gray')
        ax.set_title('DCT Reconstructed Image')
        ax.axis('off')
        self.dct_recon_canvas.draw()
        
        # Render DCT 3D plot
        self.update_dct_plot()
        self.status_label.setText('DCT visualization complete')
    
    def update_dct_plot(self):
        if self.dct_data is None:
            return
        
        # Use sampled points and precomputed transforms from worker
        real = getattr(self, '_dct_sampled_real', None)
        imag = getattr(self, '_dct_sampled_imag', None)
        z = getattr(self, '_dct_sampled_z', None)
        colors = getattr(self, '_dct_sampled_colors', None)
        limits = getattr(self, '_dct_sampled_limits', None)
        if real is None or imag is None or z is None or colors is None or limits is None:
            return
        
        title = limits.get('title', '')
        if self._dct_plot_ax is None:
            self._dct_plot_ax = self.dct_plot_figure.add_subplot(111, projection='3d')
            self._dct_scatter_artist = self._dct_plot_ax.scatter(real, imag, z, c=colors, cmap='viridis',
                                                                 marker='.', s=1, alpha=0.6)
            self._dct_plot_ax.set_xlabel('Frequency X')
            self._dct_plot_ax.set_ylabel('Frequency Y')
            self._dct_plot_ax.set_zlabel('Height')
            self._dct_plot_ax.set_title(title)
            # Set limits from worker-provided percentiles
            real_min = limits['real_min']; real_max = limits['real_max']
            imag_min = limits['imag_min']; imag_max = limits['imag_max']
            z_min = limits['z_min']; z_max = limits['z_max']
            padding = 0.1
            self._dct_plot_ax.set_xlim(real_min - padding * (real_max - real_min), real_max + padding * (real_max - real_min))
            self._dct_plot_ax.set_ylim(imag_min - padding * (imag_max - imag_min), imag_max + padding * (imag_max - imag_min))
            self._dct_plot_ax.set_zlim(z_min - padding * (z_max - z_min), z_max + padding * (z_max - z_min))
            self._dct_plot_colorbar = self.dct_plot_figure.colorbar(self._dct_scatter_artist, ax=self._dct_plot_ax,
                                                                     label=limits.get('colorbar_label', 'Magnitude'),
                                                                     shrink=0.5)
        else:
            # Update data without recreating axes
            # Update 3D positions
            self._dct_scatter_artist._offsets3d = (real, imag, z)
            # Update colors
            self._dct_scatter_artist.set_array(colors)
            self._dct_plot_ax.set_title(title)
            # Adjust limits from worker-provided bounds
            real_min = limits['real_min']; real_max = limits['real_max']
            imag_min = limits['imag_min']; imag_max = limits['imag_max']
            z_min = limits['z_min']; z_max = limits['z_max']
            padding = 0.1
            self._dct_plot_ax.set_xlim(real_min - padding * (real_max - real_min), real_max + padding * (real_max - real_min))
            self._dct_plot_ax.set_ylim(imag_min - padding * (imag_max - imag_min), imag_max + padding * (imag_max - imag_min))
            self._dct_plot_ax.set_zlim(z_min - padding * (z_max - z_min), z_max + padding * (z_max - z_min))
            # Update colorbar mapping
            if self._dct_plot_colorbar is not None:
                self._dct_plot_colorbar.update_normal(self._dct_scatter_artist)
        
        self.dct_plot_canvas.draw()
    
    def plot_psd(self):
        """Plot the Power Spectral Density of the FFT components"""
        if self.filtered_fft_data is None:
            return
        
        # Use worker-prepared display array and label to avoid main-thread compute
        psd_display = getattr(self, '_psd_display', None)
        psd_label = getattr(self, '_psd_label_cached', 'PSD')
        if psd_display is None:
            return
        
        # Create 2D PSD plot
        if self._psd_ax is None:
            self._psd_figure_ax_init(psd_display, psd_label)
        else:
            self._psd_im.set_data(psd_display)
            # Update colorbar label if changed
            if self._psd_colorbar is not None:
                try:
                    self._psd_colorbar.set_label(psd_label)
                except Exception:
                    pass
        
        self.psd_canvas.draw()

    def _psd_figure_ax_init(self, psd_display, psd_label):
        self.psd_figure.clear()
        self._psd_ax = self.psd_figure.add_subplot(111)
        self._psd_im = self._psd_ax.imshow(psd_display, cmap='hot', aspect='auto', origin='lower')
        self._psd_ax.set_title('Power Spectral Density (PSD)')
        self._psd_ax.set_xlabel('Frequency X')
        self._psd_ax.set_ylabel('Frequency Y')
        self._psd_colorbar = self.psd_figure.colorbar(self._psd_im, ax=self._psd_ax, label=psd_label)

    def apply_resize_if_needed(self):
        """Update self.image from self.image_original based on resize checkbox."""
        if self.image_original is None:
            return
        if self.resize_checkbox.isChecked():
            target_size = (128, 128)
            oh, ow = self.image_original.shape
            # Choose interpolation based on scaling direction
            interp = cv2.INTER_AREA if (oh > 128 or ow > 128) else cv2.INTER_CUBIC
            self.image = cv2.resize(self.image_original, target_size, interpolation=interp)
        else:
            self.image = self.image_original.copy()

    def add_filter_row(self):
        """Add a new filter row: [enable] [low freq] [high freq] [-]"""
        # Insert above the stretch at the end
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_widget.setLayout(row_layout)
        
        enable_checkbox = QCheckBox()
        enable_checkbox.setChecked(True)
        enable_checkbox.toggled.connect(self.on_filters_changed)
        row_layout.addWidget(enable_checkbox)
        
        # Type selector: frequency or amplitude
        type_combo = QComboBox()
        type_combo.addItems(['freq', 'amp'])
        type_combo.setFixedWidth(60)
        type_combo.currentIndexChanged.connect(self.on_filters_changed)
        row_layout.addWidget(type_combo)
        
        low_edit = QLineEdit()
        low_edit.setPlaceholderText('low')
        low_edit.setFixedWidth(70)
        low_edit.editingFinished.connect(self.on_filters_changed)
        row_layout.addWidget(low_edit)
        
        high_edit = QLineEdit()
        high_edit.setPlaceholderText('high')
        high_edit.setFixedWidth(70)
        high_edit.editingFinished.connect(self.on_filters_changed)
        row_layout.addWidget(high_edit)
        
        remove_btn = QPushButton('-')
        remove_btn.setFixedWidth(28)
        # Capture row_widget in closure for removal
        def on_remove():
            self.remove_filter_row(row_widget)
        remove_btn.clicked.connect(on_remove)
        row_layout.addWidget(remove_btn)
        
        # Keep reference
        self.filter_rows.append({
            'widget': row_widget,
            'enable': enable_checkbox,
            'type': type_combo,
            'low': low_edit,
            'high': high_edit
        })
        
        # Insert before the stretch (which is last item)
        # Remove stretch temporarily to insert before
        # Find and remove final stretch if present
        self.filters_layout.insertWidget(self.filters_layout.count() - 1, row_widget)
        #self.on_filters_changed()
    
    def remove_filter_row(self, row_widget: QWidget):
        # Remove matching entry
        self.filter_rows = [r for r in self.filter_rows if r['widget'] is not row_widget]
        # Remove widget from layout and delete it
        self.filters_layout.removeWidget(row_widget)
        row_widget.setParent(None)
        row_widget.deleteLater()
        self.on_filters_changed()
    
    def on_filters_changed(self):
        """Coalesce and enqueue compute when filters change."""
        if self.image is None:
            return
        self._enqueue_compute()
    
    def get_active_filter_ranges(self):
        """Parse enabled filters into (low, high) tuples in normalized frequency."""
        ranges = []
        for row in self.filter_rows:
            if not row['enable'].isChecked():
                continue
            low_text = row['low'].text().strip()
            high_text = row['high'].text().strip()
            if not low_text or not high_text:
                continue
            try:
                low_val = float(low_text)
                high_val = float(high_text)
            except ValueError:
                continue
            # Normalize order
            if high_val < low_val:
                low_val, high_val = high_val, low_val
            # Clamp to non-negative
            low_val = max(0.0, low_val)
            high_val = max(0.0, high_val)
            ranges.append((low_val, high_val))
        return ranges
    
    def apply_filters(self):
        """Apply active radial band-stop filters to base FFT."""
        if self.base_fft_data is None or self.frequency_radius is None:
            self.filtered_fft_data = self.base_fft_data
            # Invalidate caches
            self._cached_psd = None
            self._cached_psd_log = None
            return
        
        active_ranges = self.get_active_filter_ranges()
        if not active_ranges:
            self.filtered_fft_data = self.base_fft_data
            # Invalidate caches
            self._cached_psd = None
            self._cached_psd_log = None
            return
        
        # Create mask of ones; zero out within any active [low, high]
        mask = np.ones_like(self.base_fft_data, dtype=np.float32)
        r = self.frequency_radius
        mag = np.abs(self.base_fft_data)
        
        # Iterate actual rows to decide filter type for each (to align with active ranges order)
        # Build a list of (type, low, high) only for enabled and valid rows
        typed_ranges = []
        for row in self.filter_rows:
            if not row['enable'].isChecked():
                continue
            low_text = row['low'].text().strip()
            high_text = row['high'].text().strip()
            if not low_text or not high_text:
                continue
            try:
                low_val = float(low_text)
                high_val = float(high_text)
            except ValueError:
                continue
            if high_val < low_val:
                low_val, high_val = high_val, low_val
            low_val = max(0.0, low_val)
            high_val = max(0.0, high_val)
            typed_ranges.append((row['type'].currentText(), low_val, high_val))
        
        for kind, low, high in typed_ranges:
            if kind == 'freq':
                band = (r >= low) & (r <= high)
            else:
                band = (mag >= low) & (mag <= high)
            mask[band] = 0.0
        
        self.filtered_fft_data = self.base_fft_data * mask
        # Invalidate caches on new filtered spectrum
        self._cached_psd = None
        self._cached_psd_log = None
    
    def update_reconstruction(self):
        """Reconstruct spatial-domain image from current filtered FFT and display."""
        if self.filtered_fft_data is None:
            return
        spectrum = self.filtered_fft_data
        if self.fft_shift_checkbox.isChecked():
            spectrum = np.fft.ifftshift(spectrum)
        recon_complex = np.fft.ifft2(spectrum)
        recon = np.real(recon_complex)
        # Normalize for display to 0-255
        rmin = recon.min()
        rmax = recon.max()
        if rmax > rmin:
            recon_norm = (recon - rmin) / (rmax - rmin)
        else:
            recon_norm = np.zeros_like(recon)
        recon_img = (recon_norm * 255).astype(np.uint8)
        self.reconstructed_image = recon_img
        self.export_btn.setEnabled(True)
        
        self.recon_figure.clear()
        ax = self.recon_figure.add_subplot(111)
        ax.imshow(recon_img, cmap='gray')
        ax.set_title('Reconstructed Image')
        ax.axis('off')
        self.recon_canvas.draw()
    
    def export_image(self):
        """Export the reconstructed image, scaled to the original imported size."""
        if self.reconstructed_image is None:
            QMessageBox.information(self, 'Export', 'No reconstructed image to export yet.')
            return
        # Resize to original shape if available
        export_img = self.reconstructed_image
        if self.original_shape is not None and export_img.shape != self.original_shape:
            oh, ow = self.original_shape
            export_img = cv2.resize(export_img, (ow, oh), interpolation=cv2.INTER_CUBIC)
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Reconstructed Image', '', 'PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tiff *.tif)')
        if not save_path:
            return
        ok = cv2.imwrite(save_path, export_img)
        if ok:
            QMessageBox.information(self, 'Export', 'Image saved successfully.')
        else:
            QMessageBox.warning(self, 'Export', 'Failed to save image.')
    
    def on_resize_toggle(self):
        """Handle toggling of resize option: update processing image and recompute if needed."""
        if self.image_original is None:
            return
        self.apply_resize_if_needed()
        # Update displayed original/processing image
        self.image_figure.clear()
        ax = self.image_figure.add_subplot(111)
        ax.imshow(self.image, cmap='gray')
        ax.set_title('Original Image')
        ax.axis('off')
        self.image_canvas.draw()
        # Clear downstream products
        self.base_fft_data = None
        self.filtered_fft_data = None
        self.frequency_radius = None
        self._cached_psd = None
        self._cached_psd_log = None
        self.reconstructed_image = None
        self.recon_figure.clear()
        axr = self.recon_figure.add_subplot(111)
        axr.set_title('FFT Reconstructed Image')
        axr.axis('off')
        self.recon_canvas.draw()
        # Clear DCT data
        self.dct_data = None
        self.dct_reconstructed_image = None
        self.dct_recon_figure.clear()
        axd = self.dct_recon_figure.add_subplot(111)
        axd.set_title('DCT Reconstructed Image')
        axd.axis('off')
        self.dct_recon_canvas.draw()
        # Full recompute with new processing image
        self._enqueue_compute()
 

    def closeEvent(self, event):
        """Ensure worker thread stops and memory-heavy references are released."""
        try:
            # Stop background worker if present
            if hasattr(self, '_worker') and self._worker is not None:
                try:
                    self._worker.result_ready.disconnect(self._on_compute_result)
                except Exception:
                    pass
                try:
                    self._worker.dct_result_ready.disconnect(self._on_dct_compute_result)
                except Exception:
                    pass
                try:
                    self._worker.error.disconnect()
                except Exception:
                    pass
                try:
                    self._worker.stop()
                except Exception:
                    pass
            # Quit thread and wait
            if hasattr(self, '_thread') and self._thread is not None:
                try:
                    self._thread.quit()
                    self._thread.wait(3000)
                except Exception:
                    pass
        finally:
            # Drop references to large arrays and artists to help GC
            self.image = None
            self.image_original = None
            self.reconstructed_image = None
            self.base_fft_data = None
            self.filtered_fft_data = None
            self.frequency_radius = None
            self._cached_psd = None
            self._cached_psd_log = None
            self._sampled_real = None
            self._sampled_imag = None
            self._sampled_mag = None
            self._sampled_z = None
            self._sampled_colors = None
            self._sampled_limits = None
            self._psd_display = None
            self._psd_label_cached = None
            # Clear DCT references
            self.dct_data = None
            self.dct_reconstructed_image = None
            self._dct_sampled_real = None
            self._dct_sampled_imag = None
            self._dct_sampled_mag = None
            self._dct_sampled_z = None
            self._dct_sampled_colors = None
            self._dct_sampled_limits = None
            # Clear Matplotlib figures
            try:
                if self.plot_figure:
                    self.plot_figure.clf()
                if hasattr(self, 'dct_plot_figure') and self.dct_plot_figure:
                    self.dct_plot_figure.clf()
                if hasattr(self, 'dct_recon_figure') and self.dct_recon_figure:
                    self.dct_recon_figure.clf()
                if self.psd_figure:
                    self.psd_figure.clf()
                if self.image_figure:
                    self.image_figure.clf()
                if self.recon_figure:
                    self.recon_figure.clf()
            except Exception:
                pass
            # Null out thread/worker
            self._worker = None
            self._thread = None
            event.accept()
def main():
    app = QApplication(sys.argv)
    visualizer = FFTVisualizer()
    visualizer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()


