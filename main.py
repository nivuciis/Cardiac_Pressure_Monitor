import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from scipy.signal import butter, lfilter, find_peaks, windows
import matplotlib.pyplot as plt
import csv
import sys
from collections import deque

class BloodPressureMonitor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Setup UI
        self.setWindowTitle("Blood Pressure Monitor")
        self.resize(1000, 700)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        self.status_label = QtWidgets.QLabel("Preparing to start...")
        self.status_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.layout.addWidget(self.status_label)
        
        self.bpm_label = QtWidgets.QLabel("BPM: --")
        self.bpm_label.setStyleSheet("font-size: 32px; color: red;")
        self.layout.addWidget(self.bpm_label)
        
        self.debug_label = QtWidgets.QLabel("Debug: Waiting for data...")
        self.debug_label.setStyleSheet("font-size: 12px; color: blue;")
        self.layout.addWidget(self.debug_label)
        
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.plot_widget)
        
        self.raw_plot = self.plot_widget.addPlot(title="Cuff Pressure with Pulse Signal (mmHg)")
        self.filtered_plot = self.plot_widget.addPlot(title="Live Low-Pass Filtered Cardiac Signal & Peaks")
        
        self.raw_curve = self.raw_plot.plot(pen='y')
        self.filtered_curve = self.filtered_plot.plot(pen='g')
        self.peak_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
        self.filtered_plot.addItem(self.peak_scatter)
        
        # System parameters
        self.fs = 100  # Sampling rate (Hz)
        self.buffer_size = int(self.fs * 10) # Buffer for 10 seconds of live display
        self.data_buffer = np.zeros(self.buffer_size) 
        self.filtered_buffer = np.zeros(self.buffer_size)
        self.live_peaks = [] # For live peak display
        self.all_bpms_live = deque(maxlen=20) # Store last 20 live BPM readings for smoothing display
        
        self.start_time = QtCore.QDateTime.currentMSecsSinceEpoch()
        self.time_data = [] 
        self.raw_data = []  
        self.analysis_filtered_data = [] 
        self.analysis_peaks = [] 
        self.bpm_data_final = [] # Stores (time_index, bpm_value) from final analysis
        
        self.STATES = {
            "IDLE": 0, "INFLATING": 1, "STABLE_MAX": 2, 
            "ANALYZING": 3, "COMPLETE": 5
        }
        self.state = self.STATES["IDLE"]
        
        self.cuff_pressure = 0
        self.max_pressure = 180  # mmHg
        self.systolic_pressure = 120 # mmHg (Typical, for pulse generation window)
        self.diastolic_pressure = 60 # mmHg (Typical, for pulse generation window - as per user script)
        self.min_pressure = 0  # mmHg - Deflate to zero
        self.inflation_rate = 30  # mmHg/s
        self.deflation_rate = 2.5  # mmHg/s
        
        self.heart_rate_for_simulation = 100  # BPM, for generating synthetic pulse
        self.pulse_amp_scale = 1.2 # Amplitude scale for the synthetic pulse
        self.pulse_sig_retf = None
        self.pulse_signal_idx = 0
        
        self.b_low, self.a_low = self.butter_lowpass(3, self.fs, order=4) # Lowered order for less ringing
        self.b_high, self.a_high = self.butter_highpass(0.5, self.fs, order=4) # High-pass filter for baseline removal
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000/self.fs)) 
        
        QtCore.QTimer.singleShot(1000, self.start_inflation)
    
    def butter_lowpass(self, cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    def butter_highpass(self, cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    def start_inflation(self):
        self.state = self.STATES["INFLATING"]
        self.status_label.setText("Inflating cuff...")
        self.inflation_start_time = QtCore.QDateTime.currentMSecsSinceEpoch()
    
    def start_analysis_phase(self):
        self.state = self.STATES["ANALYZING"]
        self.status_label.setText("Deflating and collecting data...")
        self.analysis_start_time = QtCore.QDateTime.currentMSecsSinceEpoch()
        
        fhr = self.heart_rate_for_simulation / 60.0
        fwhr = 2 * np.pi * fhr
        
        # Time window where pulses are active (between diastolic and systolic)
        pulse_active_duration = (self.systolic_pressure - self.diastolic_pressure) / self.deflation_rate
        if pulse_active_duration <= 0:
            pulse_active_duration = 15 # Default if pressures are unusual (e.g. dia > sys)

        # Create time vector for the pulse generation
        # Ensure it spans the duration where pulses should appear
        num_pulse_samples = int(self.fs * pulse_active_duration)
        if num_pulse_samples <= 0:
            self.pulse_sig_retf = None
            print("Warning: Pulse generation time is too short or zero. No pulse will be added.")
            return

        pulse_t_local = np.linspace(0, pulse_active_duration, num_pulse_samples, endpoint=False)
        
        # Gaussian window for amplitude modulation of pulses
        # std is a fraction of the number of samples in pulse_t_local
        std_gaussian_factor = 4.0 # Adjust this to change the width of the Gaussian envelope
        std_gaussian = num_pulse_samples / std_gaussian_factor
        if std_gaussian < 1: std_gaussian = 1 # Ensure std is at least 1

        g_filt = windows.gaussian(num_pulse_samples, std_gaussian)
        
        # Base sinusoidal pulse
        pulse = np.sin(fwhr * pulse_t_local) 
        
        # Modulate pulse with Gaussian window
        pulse_sig_modulated = pulse * g_filt
        
        # Rectify: take only positive parts, as physiological pulses are usually seen as pressure increases
        self.pulse_sig_retf = np.array([p if p > 0 else 0 for p in pulse_sig_modulated])
        self.pulse_signal_idx = 0
        
        self.debug_label.setText(f"Debug: Pulse signal generated ({len(self.pulse_sig_retf)} samples).")

    def update(self):
        current_time_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
        current_elapsed_s = (current_time_ms - self.start_time) / 1000.0
        
        noise_amplitude = 0.2 # Reduced noise
        current_cuff_pressure_sample = self.cuff_pressure 
        
        if self.state == self.STATES["INFLATING"]:
            elapsed_inflation_s = (current_time_ms - self.inflation_start_time) / 1000.0
            self.cuff_pressure = min(self.max_pressure, self.inflation_rate * elapsed_inflation_s)
            current_cuff_pressure_sample = self.cuff_pressure + noise_amplitude * np.random.randn()
            
            if self.cuff_pressure >= self.max_pressure:
                self.state = self.STATES["STABLE_MAX"]
                self.status_label.setText("Max pressure reached. Deflating soon.")
                QtCore.QTimer.singleShot(1000, self.start_analysis_phase) 
        
        elif self.state == self.STATES["ANALYZING"]:
            elapsed_analysis_s = (current_time_ms - self.analysis_start_time) / 1000.0
            self.cuff_pressure = max(self.min_pressure, self.max_pressure - elapsed_analysis_s * self.deflation_rate)
            current_cuff_pressure_sample = self.cuff_pressure + noise_amplitude * np.random.randn()
            
            if self.pulse_sig_retf is not None and \
               self.diastolic_pressure < self.cuff_pressure < self.systolic_pressure + 10: # Window for adding pulses
                if self.pulse_signal_idx < len(self.pulse_sig_retf):
                    current_cuff_pressure_sample += self.pulse_amp_scale * self.pulse_sig_retf[self.pulse_signal_idx]
                    self.pulse_signal_idx += 1
            
            self.debug_label.setText(f"Debug: P: {self.cuff_pressure:.1f} mmHg, Pulses: {self.pulse_signal_idx}/{len(self.pulse_sig_retf) if self.pulse_sig_retf is not None else 'N/A'}")

            if self.cuff_pressure <= self.min_pressure:
                self.state = self.STATES["COMPLETE"]
                self.status_label.setText("Data collection complete. Analyzing...")
                QtCore.QTimer.singleShot(100, self.calculate_final_bpm) 
        
        elif self.state == self.STATES["COMPLETE"]:
            self.timer.stop()
            return

        self.data_buffer = np.roll(self.data_buffer, -1)
        self.data_buffer[-1] = current_cuff_pressure_sample
        
        self.time_data.append(current_elapsed_s)
        self.raw_data.append(current_cuff_pressure_sample)
        
        if len(self.raw_data) > len(self.b_low): 
            # Apply low-pass then high-pass filter for live display
            self.filtered_buffer = lfilter(self.b_low, self.a_low, self.data_buffer)
            self.filtered_buffer = lfilter(self.b_high, self.a_high, self.filtered_buffer)

            # Live peak detection on filtered_buffer for display
            peak_height_live = 0.1 * self.pulse_amp_scale 
            prominence_live = 0.05 * self.pulse_amp_scale
            distance_live = int(self.fs * 0.3) # Max ~200 BPM

            # Use a more stable segment of the filtered buffer for live peak detection
            # Avoids issues at the very start of the buffer
            detect_window = min(len(self.filtered_buffer), self.fs * 3) # Last 3s or less
            signal_for_live_peaks = self.filtered_buffer[-detect_window:]
            
            # Detrend or normalize if necessary, but direct detection might work if pulses are clear
            # For simplicity, direct detection on the low-pass filtered signal.
            # Consider that filtered_buffer still contains the deflation ramp.
            # A simple way to make peaks more apparent is to subtract a very smoothed version or mean.
            # However, find_peaks with prominence can often handle this.
            # Let's try to detect on the signal as is, relying on pulse_amp_scale and prominence.

            self.live_peaks, _ = find_peaks(signal_for_live_peaks, 
                                             height=peak_height_live,
                                             prominence=prominence_live,
                                             distance=distance_live)
            # Adjust peak indices to match the full filtered_buffer
            self.live_peaks += (len(self.filtered_buffer) - detect_window)


            if len(self.live_peaks) > 1:
                rr_intervals_live = np.diff(self.live_peaks) / self.fs
                valid_intervals_live = rr_intervals_live[(rr_intervals_live > 0.3) & (rr_intervals_live < 2.0)] # 30-200 BPM
                if len(valid_intervals_live) > 0:
                    current_bpm_live = 60 / np.mean(valid_intervals_live)
                    self.all_bpms_live.append(current_bpm_live)
                    if len(self.all_bpms_live) > 0: # Check if deque is not empty
                         # smoothed_bpm_live = np.median(list(self.all_bpms_live)) # Median of recent BPMs
                         # If using median, ensure there are enough samples for stability.
                         # For now, just use the latest valid BPM.
                        self.bpm_label.setText(f"BPM: {current_bpm_live:.1f} (Live)")
                    else:
                        self.bpm_label.setText(f"BPM: -- (Live)")
            # Update live peak scatter plot
            if len(self.live_peaks) > 0:
                peak_values_live = self.filtered_buffer[self.live_peaks]
                #Detect pressure when signal starts and ends
                print(f"Live Peaks: {len(self.live_peaks)}, Values: {peak_values_live}")
                self.peak_scatter.setData(self.live_peaks, peak_values_live)
            else:
                self.peak_scatter.clear()
        
        self.raw_curve.setData(self.data_buffer)
        self.filtered_curve.setData(self.filtered_buffer)
        
        QtWidgets.QApplication.processEvents()
    
    def calculate_final_bpm(self):
        self.status_label.setText("Performing final BPM analysis...")
        if not self.raw_data or len(self.raw_data) < max(len(self.a_low), len(self.b_low)):
            self.bpm_label.setText("Final BPM: No/Not enough data")
            self.status_label.setText("Error: No or insufficient data collected.")
            return

        self.analysis_filtered_data = lfilter(self.b_low, self.a_low, self.raw_data)
        self.analysis_filtered_data = lfilter(self.b_high, self.a_high, self.analysis_filtered_data)
        
        # Detrend or remove baseline from analysis_filtered_data to improve peak detection
        # A simple approach: subtract the mean. For signals with a strong trend (like deflation),
        # a more robust detrending (e.g., subtracting a polynomial fit or a heavily smoothed version)
        # might be better, but let's try with mean removal first, relying on prominence.
        signal_for_final_peaks = self.analysis_filtered_data - np.mean(self.analysis_filtered_data)
        
        # Adjust peak detection parameters for the final analysis
        peak_height_final = 0.2 * self.pulse_amp_scale  # Expect clearer peaks in full signal
        prominence_final = 0.1 * self.pulse_amp_scale
        distance_final = int(self.fs * 0.3) # Max ~200 BPM, min ~0.3s interval

        self.analysis_peaks, _ = find_peaks(signal_for_final_peaks, 
                                            height=peak_height_final,
                                            prominence=prominence_final,
                                            distance=distance_final)
        
        final_bpms_calculated = []
        self.bpm_data_final = []

        if len(self.analysis_peaks) > 1:
            rr_intervals_samples = np.diff(self.analysis_peaks)
            rr_intervals_s = rr_intervals_samples / self.fs
            valid_intervals_s = rr_intervals_s[(rr_intervals_s > 0.3) & (rr_intervals_s < 2.0)] 
            
            # --- Extract Systolic and Diastolic from signal ---
            # Get the cuff pressures at each detected peak
            peak_pressures = np.array(self.raw_data)[self.analysis_peaks]
            # Systolic: pressure at first valid peak
            systolic_from_signal = peak_pressures[0]
            # Diastolic: pressure at last valid peak
            diastolic_from_signal = peak_pressures[-1]
            # Show on status/debug
            self.status_label.setText(
                f"Analysis Complete. Systolic: {systolic_from_signal:.1f} mmHg, Diastolic: {diastolic_from_signal:.1f} mmHg"
            )
            self.debug_label.setText(
                self.debug_label.text() + 
                f" | Systolic: {systolic_from_signal:.1f} | Diastolic: {diastolic_from_signal:.1f}"
            )
            # Optionally, store for later use
            self.systolic_from_signal = systolic_from_signal
            self.diastolic_from_signal = diastolic_from_signal
            # --- End extraction ---

            if len(valid_intervals_s) > 0:
                final_bpms_calculated = 60 / valid_intervals_s
                median_bpm_final = np.median(final_bpms_calculated)
                self.bpm_label.setText(f"Final BPM: {median_bpm_final:.1f}")
                self.status_label.setText("Analysis Complete.")
                self.debug_label.setText(f"Debug: Final Peaks: {len(self.analysis_peaks)}, Median BPM: {median_bpm_final:.1f}")

                # Populate bpm_data_final for CSV
                for i in range(len(final_bpms_calculated)):
                    # Assign BPM to the time of the second peak of the interval
                    peak_index_of_bpm = self.analysis_peaks[i+1] 
                    if peak_index_of_bpm < len(self.time_data):
                        self.bpm_data_final.append((peak_index_of_bpm, final_bpms_calculated[i]))
            else:
                self.bpm_label.setText("Final BPM: N/A (intervals out of range)")
                self.status_label.setText("No valid heart rate intervals in final analysis.")
        else:
            self.bpm_label.setText("Final BPM: N/A (not enough peaks)")
            self.status_label.setText("Not enough peaks for final BPM analysis.")

        try:
            csv_file = self.save_data_to_csv()
            print(f"Data saved to {csv_file}")
            self.plot_results()
        except Exception as e:
            print(f"Error during data saving/plotting: {str(e)}")
            self.status_label.setText(f"Error saving/plotting: {str(e)}")
    
    def save_data_to_csv(self):
        filename = "blood_pressure_data_final_analysis.csv"
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time (s)", "Raw Pressure (mmHg)", "Low-Pass Filtered (Amplitude)", "BPM (at peak time)"])
            
            bpm_dict = {idx: val for idx, val in self.bpm_data_final} 

            for i in range(len(self.time_data)):
                bpm_val = bpm_dict.get(i, "") 
                writer.writerow([
                    f"{self.time_data[i]:.3f}",
                    f"{self.raw_data[i]:.2f}",
                    f"{self.analysis_filtered_data[i]:.4f}" if i < len(self.analysis_filtered_data) else "",
                    f"{bpm_val:.1f}" if bpm_val else ""
                ])
        return filename
    
    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(self.time_data, self.raw_data, 'b-', label='Raw Cuff Pressure')
        ax1.set_title('Cuff Pressure (Raw Data)')
        ax1.set_ylabel('Pressure (mmHg)')
        ax1.grid(True)
        ax1.legend()
        
        # --- Systolic/Diastolic detection based on analysis phase only ---
        # Find the index in time_data where analysis started
        analysis_start_time_s = (self.analysis_start_time - self.start_time) / 1000.0
        analysis_start_idx = next((i for i, t in enumerate(self.time_data) if t >= analysis_start_time_s), 0)
        valid_peaks_plot = [p for p in self.analysis_peaks if p >= analysis_start_idx and p < len(self.time_data) and p < len(self.analysis_filtered_data)]
        systolic_idx = valid_peaks_plot[2] if valid_peaks_plot else None
        diastolic_idx = valid_peaks_plot[-1] if valid_peaks_plot else None

        if self.analysis_filtered_data is not None and len(self.analysis_filtered_data) > 0:
            ax2.plot(self.time_data, self.analysis_filtered_data, 'g-', label='Low-Pass Filtered Signal')
            if valid_peaks_plot:
                ax2.plot(np.array(self.time_data)[valid_peaks_plot], 
                         np.array(self.analysis_filtered_data)[valid_peaks_plot], 
                         "ro", markersize=5, label='Detected Peaks')
                # Mark Systolic and Diastolic on ax1 (raw pressure)
                if systolic_idx is not None:
                    ax1.plot(self.time_data[systolic_idx], self.raw_data[systolic_idx], "go", markersize=10, label='Systolic (from signal)')
                    ax1.annotate(f"Systolic\n{self.raw_data[systolic_idx]:.1f} mmHg", 
                                 (self.time_data[systolic_idx], self.raw_data[systolic_idx]),
                                 textcoords="offset points", xytext=(0,10), ha='center', color='green', fontsize=10)
                if diastolic_idx is not None:
                    ax1.plot(self.time_data[diastolic_idx], self.raw_data[diastolic_idx], "mo", markersize=10, label='Diastolic (from signal)')
                    ax1.annotate(f"Diastolic\n{self.raw_data[diastolic_idx]:.1f} mmHg", 
                                 (self.time_data[diastolic_idx], self.raw_data[diastolic_idx]),
                                 textcoords="offset points", xytext=(0,-20), ha='center', color='purple', fontsize=10)
                ax1.legend()
            ax2.set_title('Low-Pass Filtered Signal & Detected Peaks')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Filtered Amplitude')
            ax2.grid(True)
            ax2.legend()
        else:
            ax2.set_title('Low-Pass Filtered Signal (No data or error)')
        
        final_bpm_text = self.bpm_label.text().replace("Final BPM: ", "") # Get from label
        plt.suptitle(f"Blood Pressure Analysis - Final BPM: {final_bpm_text}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv) 
    monitor = BloodPressureMonitor()
    monitor.show()
    sys.exit(app.exec_())