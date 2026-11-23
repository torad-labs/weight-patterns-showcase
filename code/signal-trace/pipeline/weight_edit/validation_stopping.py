"""Finite checkpoint selection from fresh fitted loss and a separate, frozen development panel."""
import math
from statistics import median


def numeric(x):
    if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
        raise ValueError('expected a finite nonnegative measurement')
    return x


def spread(values):
    if len(values) < 2:
        raise ValueError('unchanged-weight replay needs at least two observations')
    return max(map(numeric, values)) - min(map(numeric, values))


def recovered_delay(losses):
    """One more update than the longest observed dry spell ending in a new low."""
    if len(losses) < 3:
        raise ValueError('the calibration trajectory needs successive installed measurements')
    best = numeric(losses[0])
    longest = since = 0
    recovered = False
    for x in losses[1:]:
        if numeric(x) < best:
            longest = max(longest, since)
            recovered |= since > 0
            best, since = x, 0
        else:
            since += 1
    if not recovered:
        raise ValueError('no recovered reversal to calibrate validation cadence')
    return longest + 1


class Policy:
    def __init__(self, calibration, operations):
        self.operations = operations
        if not operations or set(operations) != set(calibration['operation']):
            raise ValueError('calibrate every declared operation')
        self.fit_resolution = spread(calibration['fit_repeat'])
        self.cadence = recovered_delay(calibration['fit_history'])
        history = calibration['fit_history']
        window = self.cadence
        adjacent = [median(history[i-window:i]) - median(history[i:i+window])
                    for i in range(window, len(history)-window+1)]
        # Empirical dispersion in changes across actual training windows is
        # distinct from unchanged-checkpoint replay precision.
        center = median(adjacent) if adjacent else 0.
        self.fit_variation = median(abs(x-center) for x in adjacent) if adjacent else 0.
        # The calibration run is a declared, finite compute allowance. It is
        # separate from the scientific reasons for ending early.
        self.budget = len(calibration['fit_history']) - 1
        self.bands = {}
        for name, cfg in operations.items():
            observed = calibration['operation'][name]
            if not observed['qualified_correct'] or cfg['role'] not in ('repair', 'retain'):
                raise ValueError('each operation needs a qualified correct range and a role')
            correct = list(map(numeric, observed['qualified_correct']))
            self.bands[name] = (max(correct), spread(observed['repeat']), max(correct)-min(correct))
        self.fit = []
        self.panels = []
        self.selected = None

    def due(self, n):
        return n % self.cadence == 0 or n == self.budget

    def observe(self, n, loss, panel=None):
        if n != len(self.fit) or n > self.budget or (self.panels and self.panels[-1]['terminal']):
            raise ValueError('each complete installed generation must appear once in order')
        self.fit.append(numeric(loss))
        if not self.due(n):
            if panel is not None:
                raise ValueError('undeclared validation checkpoint')
            return None
        if panel is None or set(panel) != set(self.operations):
            raise ValueError('validation must account for every operation')
        good, right = set(), set()
        unavailable = []
        scores = {}
        for name, row in panel.items():
            if row['status'] not in ('right', 'wrong', 'absent', 'unscorable'):
                raise ValueError(f'unknown operation status for {name}')
            if row['status'] == 'right':
                right.add(name)
            if row['status'] in ('absent', 'unscorable') or row['distance'] is None:
                unavailable.append(name)
                continue
            scores[name] = numeric(row['distance'])
            high, resolution, _ = self.bands[name]
            if row['status'] == 'right' and scores[name] <= high + resolution:
                good.add(name)
        required = {k for k, cfg in self.operations.items() if cfg['role'] == 'retain'}
        previous = self.panels[-1] if self.panels else None
        row = dict(generation=n, fitted_loss=loss, operations=panel, scores=scores, qualified=sorted(good),
                   arithmetic_right=sorted(right), right_outside_geometry=sorted(right-good),
                   unavailable=unavailable, fit_window_gain=None, validation_gain=None,
                   fit_replay_resolution=self.fit_resolution, fit_window_variation=self.fit_variation,
                   promote=False, decision='continue', terminal=False)
        self.panels.append(row)
        if unavailable or (n == 0 and not required <= good):
            row['decision'] = ('incomplete_validation' if unavailable else 'starting_controls_incorrect'
                               if not required <= right else 'starting_control_geometry_outside_calibration')
        else:
            incumbent = set(self.selected['qualified']) if self.selected else set()
            prior_right = set(self.selected['arithmetic_right']) if self.selected else set()
            eligible = required <= good and incumbent <= good and prior_right <= right
            if eligible and (self.selected is None or good > incumbent or right > prior_right):
                self.selected = row
                row['promote'] = True
            if len(good) == len(self.operations):
                row['decision'] = 'first_observed_joint_panel_closure'
            elif previous is not None:
                old_good = set(previous['qualified'])
                distances_improved = [k for k in scores if k not in good and
                                      previous['scores'][k] - scores[k] > max(self.bands[k][1:])]
                old_right = set(previous['arithmetic_right'])
                row['validation_gain'] = dict(newly_qualified=sorted(good-old_good),
                                              newly_right=sorted(right-old_right),closer=distances_improved)
                # Assess only two full observed update windows. An isolated
                # rebound or a short final partial window cannot establish a plateau.
                if n >= 2*self.cadence and n % self.cadence == 0:
                    h = self.cadence
                    row['fit_window_gain'] = median(self.fit[-2*h:-h]) - median(self.fit[-h:])
                    two_back = self.panels[-3]
                    last_gain = any(previous['validation_gain'].values())
                    # A loss of a previously acquired operation must occur at
                    # two consecutive scheduled checkpoints before it ends the run.
                    lost = incumbent-good
                    if (prior_right-right) & (prior_right-old_right):
                        row['decision'] = 'persistent_arithmetic_regression'
                    elif lost & (incumbent-old_good):
                        row['decision'] = 'persistent_trace_departure'
                    elif (not last_gain and not (good-set(two_back['qualified']))
                          and not any(row['validation_gain'].values())):
                        row['decision'] = ('unresolved_fit_and_validation_gain'
                                           if row['fit_window_gain'] <= max(self.fit_resolution,self.fit_variation)
                                           else 'fitted_gain_without_validation_gain')
            if n == self.budget and row['decision'] == 'continue':
                row['decision'] = 'compute_allowance_exhausted'
        row['terminal'] = row['decision'] != 'continue'
        row['selected_generation'] = self.selected['generation'] if self.selected else None
        return row
