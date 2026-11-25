/* Campus Conflict Perception Console - Motion & Visualization Engine
 * Pure classic browser script (no build step, no external dependencies).
 * Exports window.CampusMotion = {
 *   RollingCounter,
 *   AnimatedSignalPipeline,
 *   CampusRoadTwinCanvas,
 *   SpeedTimelineChart,
 *   CalibrationCanvas,
 *   AlertAudioSynth,
 *   SimulatedCameraCanvas,
 *   _buildScenarioState,
 *   _computeCpaAndTtc,
 *   _computeCvmCircleTtc
 * }
 */
(function (global) {
    "use strict";

    var RISK_NAMES = [
        "LEVEL_0_SAFE",
        "LEVEL_1_SPEEDING",
        "LEVEL_2_CAUTION",
        "LEVEL_3_CRITICAL"
    ];

    var VEHICLE_DIMS = {
        car: { lengthM: 4.5, widthM: 1.9 },
        suv: { lengthM: 4.7, widthM: 1.95 },
        motor_vehicle: { lengthM: 4.5, widthM: 1.9 },
        vehicle: { lengthM: 4.5, widthM: 1.9 },
        bus: { lengthM: 9.6, widthM: 2.5 },
        truck: { lengthM: 7.6, widthM: 2.45 },
        motorcycle: { lengthM: 2.1, widthM: 0.8 },
        bicycle: { lengthM: 1.8, widthM: 0.7 },
        pedestrian: { lengthM: 0.7, widthM: 0.7 },
        person: { lengthM: 0.7, widthM: 0.7 }
    };

    function clamp(val, min, max) {
        return val < min ? min : (val > max ? max : val);
    }

    function isFiniteNum(v) {
        return typeof v === "number" && Number.isFinite(v);
    }

    function toNum(v, fallback) {
        if (v === null || v === undefined || v === "" || typeof v === "boolean") {
            return fallback;
        }
        var n = Number(v);
        return Number.isFinite(n) ? n : fallback;
    }

    function parseRiskLevel(raw) {
        if (typeof raw === "number" && Number.isFinite(raw)) {
            return clamp(Math.round(raw), 0, 3);
        }
        if (typeof raw === "string") {
            var s = raw.trim().toUpperCase();
            if (s === "LEVEL_3_CRITICAL" || s === "CRITICAL" || s === "3") return 3;
            if (s === "LEVEL_2_CAUTION" || s === "CAUTION" || s === "2") return 2;
            if (s === "LEVEL_1_SPEEDING" || s === "SPEEDING" || s === "1") return 1;
            if (s === "LEVEL_0_SAFE" || s === "SAFE" || s === "0") return 0;
            var n = Number(s);
            if (Number.isFinite(n)) return clamp(Math.round(n), 0, 3);
        }
        return 0;
    }

    /* ----------------------------------------------------------------------
     * Reduced-motion media tracker (WCAG 2.2.2 / 2.3.1)
     * ---------------------------------------------------------------------- */
    function createReducedMotionTracker(onChange) {
        var mql = (typeof global.matchMedia === "function")
            ? global.matchMedia("(prefers-reduced-motion: reduce)")
            : null;
        var reduced = mql ? Boolean(mql.matches) : false;

        function handleChange(ev) {
            reduced = Boolean(ev && ev.matches);
            if (typeof onChange === "function") {
                onChange(reduced);
            }
        }

        if (mql) {
            if (typeof mql.addEventListener === "function") {
                mql.addEventListener("change", handleChange);
            } else if (typeof mql.addListener === "function") {
                mql.addListener(handleChange);
            }
        }

        return {
            isReduced: function () {
                return reduced;
            },
            destroy: function () {
                if (!mql) return;
                if (typeof mql.removeEventListener === "function") {
                    mql.removeEventListener("change", handleChange);
                } else if (typeof mql.removeListener === "function") {
                    mql.removeListener(handleChange);
                }
            }
        };
    }

    /* ----------------------------------------------------------------------
     * Dynamic CSS custom property color reader & light-dark() evaluator
     * ---------------------------------------------------------------------- */
    function createColorProbe() {
        if (typeof document === "undefined" || !document.documentElement) {
            return null;
        }
        var el = document.createElement("span");
        el.setAttribute("aria-hidden", "true");
        el.style.position = "absolute";
        el.style.width = "0";
        el.style.height = "0";
        el.style.overflow = "hidden";
        el.style.pointerEvents = "none";
        el.style.visibility = "hidden";
        (document.body || document.documentElement).appendChild(el);
        return el;
    }

    function resolveCssColor(probe, rootStyle, varNames, fallback) {
        if (!rootStyle) return fallback;
        for (var i = 0; i < varNames.length; i += 1) {
            var raw = rootStyle.getPropertyValue(varNames[i]);
            if (!raw) continue;
            var trimmed = raw.trim();
            if (!trimmed) continue;
            if (probe && typeof global.getComputedStyle === "function") {
                // Evaluate CSS expressions such as light-dark(...) or color-mix(...)
                probe.style.color = "";
                probe.style.color = "var(" + varNames[i] + ")";
                if (probe.style.color) {
                    var computed = global.getComputedStyle(probe).color;
                    if (computed) return computed;
                }
            }
            if (trimmed.indexOf("light-dark(") === -1 && trimmed.indexOf("var(") === -1) {
                return trimmed;
            }
        }
        return fallback;
    }

    function parseRgbChannels(colorStr) {
        if (typeof colorStr !== "string") return null;
        var s = colorStr.trim();
        if (s.charAt(0) === "#") {
            var hex = s.slice(1);
            if (hex.length === 3 || hex.length === 4) {
                var r3 = parseInt(hex.charAt(0) + hex.charAt(0), 16);
                var g3 = parseInt(hex.charAt(1) + hex.charAt(1), 16);
                var b3 = parseInt(hex.charAt(2) + hex.charAt(2), 16);
                return Number.isFinite(r3) ? [r3, g3, b3] : null;
            }
            if (hex.length === 6 || hex.length === 8) {
                var r6 = parseInt(hex.slice(0, 2), 16);
                var g6 = parseInt(hex.slice(2, 4), 16);
                var b6 = parseInt(hex.slice(4, 6), 16);
                return Number.isFinite(r6) ? [r6, g6, b6] : null;
            }
        }
        var match = s.match(/rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i);
        if (match) {
            return [Number(match[1]), Number(match[2]), Number(match[3])];
        }
        return null;
    }

    function withAlpha(colorStr, alpha) {
        var a = clamp(alpha, 0, 1);
        var ch = parseRgbChannels(colorStr);
        if (!ch) return colorStr;
        return "rgba(" + Math.round(ch[0]) + ", " + Math.round(ch[1]) + ", " + Math.round(ch[2]) + ", " + a.toFixed(3) + ")";
    }

    function detectLightTheme(rootStyle) {
        if (typeof document === "undefined" || !document.documentElement) {
            return false;
        }
        var root = document.documentElement;
        var attr = (root.getAttribute("data-theme") || "").toLowerCase();
        if (attr === "light") return true;
        if (attr === "dark") return false;
        if (root.classList && root.classList.contains("light")) return true;
        if (rootStyle && rootStyle.colorScheme) {
            var cs = rootStyle.colorScheme.trim().toLowerCase();
            if (cs === "light") return true;
            if (cs === "dark") return false;
        }
        return false;
    }

    function readThemePalette(probe) {
        var rootStyle = (typeof document !== "undefined" && document.documentElement && typeof global.getComputedStyle === "function")
            ? global.getComputedStyle(document.documentElement)
            : null;
        var isLight = detectLightTheme(rootStyle);

        var d = isLight
            ? {
                bgPage: "#fafafb",
                bgCard: "#ffffff",
                bgRaised: "#f4f5f7",
                roadShoulder: "#eff1f5",
                roadSurface: "#e6e9ef",
                roadMark: "rgba(20, 21, 24, 0.22)",
                roadCurb: "rgba(20, 21, 24, 0.30)",
                gridLine: "rgba(20, 21, 24, 0.08)",
                hairline: "rgba(20, 21, 24, 0.12)",
                textPrimary: "#141518",
                textSecondary: "#4b4e57",
                textMuted: "#6f737d",
                accent: "#b47400",
                safe: "#1d8f5a",
                critical: "#c93136",
                neutralPipe: "#8c929c",
                vehicleNeutral: "#2b3340",
                parkingZone: "#8f6716"
            }
            : {
                bgPage: "#141415",
                bgCard: "#19191b",
                bgRaised: "#232325",
                roadShoulder: "#111214",
                roadSurface: "#17181c",
                roadMark: "rgba(237, 237, 237, 0.20)",
                roadCurb: "rgba(237, 237, 237, 0.26)",
                gridLine: "rgba(237, 237, 237, 0.065)",
                hairline: "rgba(237, 237, 237, 0.10)",
                textPrimary: "#ededed",
                textSecondary: "#a1a1a6",
                textMuted: "#85858b",
                accent: "#e5a00d",
                safe: "#4cc38a",
                critical: "#e5484d",
                neutralPipe: "#5d626d",
                vehicleNeutral: "#8a94a6",
                parkingZone: "#c8962e"
            };

        return {
            isLight: isLight,
            bgPage: resolveCssColor(probe, rootStyle, ["--bg-page", "--bg-canvas", "--surface-page", "--bg"], d.bgPage),
            bgCard: resolveCssColor(probe, rootStyle, ["--bg-card", "--surface-card", "--card-bg", "--bg-surface"], d.bgCard),
            bgRaised: resolveCssColor(probe, rootStyle, ["--bg-raised", "--surface-raised", "--bg-elevated", "--bg-subtle"], d.bgRaised),
            roadShoulder: resolveCssColor(probe, rootStyle, ["--road-shoulder", "--bg-sidebar"], d.roadShoulder),
            roadSurface: resolveCssColor(probe, rootStyle, ["--road-surface", "--bg-card"], d.roadSurface),
            roadMark: d.roadMark,
            roadCurb: d.roadCurb,
            gridLine: d.gridLine,
            hairline: resolveCssColor(probe, rootStyle, ["--border-hairline", "--hairline", "--border-subtle", "--border"], d.hairline),
            textPrimary: resolveCssColor(probe, rootStyle, ["--text-primary", "--text", "--fg-primary", "--ink"], d.textPrimary),
            textSecondary: resolveCssColor(probe, rootStyle, ["--text-secondary", "--fg-secondary"], d.textSecondary),
            textMuted: resolveCssColor(probe, rootStyle, ["--text-muted", "--fg-muted", "--muted"], d.textMuted),
            accent: resolveCssColor(probe, rootStyle, ["--accent-amber", "--accent", "--color-accent", "--amber"], d.accent),
            safe: resolveCssColor(probe, rootStyle, ["--semantic-safe", "--color-safe", "--safe", "--status-safe"], d.safe),
            critical: resolveCssColor(probe, rootStyle, ["--semantic-critical", "--color-critical", "--danger", "--critical", "--status-critical"], d.critical),
            neutralPipe: d.neutralPipe,
            vehicleNeutral: d.vehicleNeutral,
            parkingZone: d.parkingZone
        };
    }

    function observeThemeChanges(onThemeChange) {
        if (typeof MutationObserver !== "function" || typeof document === "undefined" || !document.documentElement) {
            return function () {};
        }
        var obs = new MutationObserver(function () {
            onThemeChange();
        });
        obs.observe(document.documentElement, {
            attributes: true,
            attributeFilter: ["data-theme", "class", "style"]
        });
        return function () {
            obs.disconnect();
        };
    }

    function riskColor(palette, riskLevel) {
        var lvl = parseRiskLevel(riskLevel);
        if (lvl === 3) return palette.critical;
        if (lvl === 2 || lvl === 1) return palette.accent;
        return palette.safe;
    }

    function attachResizeObserver(targetEl, instance, callback) {
        instance._resizeRafId = null;
        if (typeof ResizeObserver !== "function" || !targetEl) {
            instance._resizeObserver = null;
            return null;
        }
        var ro = new ResizeObserver(function () {
            if (instance._destroyed) return;
            if (typeof global.requestAnimationFrame === "function") {
                if (instance._resizeRafId !== null) return;
                instance._resizeRafId = global.requestAnimationFrame(function () {
                    instance._resizeRafId = null;
                    if (!instance._destroyed) {
                        callback();
                    }
                });
            } else {
                callback();
            }
        });
        ro.observe(targetEl);
        instance._resizeObserver = ro;
        return ro;
    }

    function detachResizeObserver(instance) {
        if (instance._resizeRafId !== null && typeof global.cancelAnimationFrame === "function") {
            global.cancelAnimationFrame(instance._resizeRafId);
            instance._resizeRafId = null;
        }
        if (instance._resizeObserver) {
            instance._resizeObserver.disconnect();
            instance._resizeObserver = null;
        }
    }

    /* ======================================================================
     * 1. RollingCounter (persistent slot reuse, zero DOM thrash on 100ms ticks)
     * ====================================================================== */
    function RollingCounter(el, options) {
        if (!(this instanceof RollingCounter)) {
            return new RollingCounter(el, options);
        }
        if (!el) {
            throw new Error("RollingCounter requires a valid DOM element");
        }
        var opts = options || {};
        this._el = el;
        this._decimals = clamp(toNum(opts.decimals, 1), 0, 6);
        this._unit = typeof opts.unit === "string" ? opts.unit : "";
        this._fallback = typeof opts.fallback === "string" ? opts.fallback : "--";
        this._slots = [];
        this._unitEl = null;
        this._templateKey = null;
        this._lastDigits = [];
        this._lastNumText = null;
        this._lastValid = null;
        this._lastUnit = null;
        this._lastUpdateMs = 0;
        this._lastDurationMs = 180;
        this._rafId = null;
        this._destroyed = false;

        var self = this;
        this._motionTracker = createReducedMotionTracker(function (isReduced) {
            self._applyReducedState(isReduced);
        });

        this._el.classList.add("nf-counter");
        this._el.style.display = "inline-flex";
        this._el.style.alignItems = "baseline";
        this._el.style.fontVariantNumeric = "tabular-nums";
        this._el.style.whiteSpace = "nowrap";

        this.setValue(null);
    }

    RollingCounter.prototype._applyReducedState = function (isReduced) {
        var dur = this._lastDurationMs || 180;
        var transitionVal = isReduced
            ? "none"
            : "transform " + dur + "ms cubic-bezier(0.22, 1, 0.36, 1)";
        for (var i = 0; i < this._slots.length; i += 1) {
            if (this._slots[i].kind === "digit" && this._slots[i].strip) {
                this._slots[i].strip.style.transition = transitionVal;
                if (isReduced) {
                    this._slots[i].strip.style.transform = "translateY(-" + (this._slots[i].targetDigit * 10) + "%)";
                    this._slots[i].currentDigit = this._slots[i].targetDigit;
                }
            }
        }
    };

    RollingCounter.prototype._createDigitSlot = function (initialDigit, targetDigit, isReduced, snapNow) {
        var slot = document.createElement("span");
        slot.className = "nf-slot";
        slot.setAttribute("aria-hidden", "true");
        slot.style.display = "inline-block";
        slot.style.overflow = "hidden";
        slot.style.height = "1.25em";
        slot.style.lineHeight = "1.25em";
        slot.style.verticalAlign = "baseline";
        slot.style.width = "1ch";
        slot.style.textAlign = "center";

        var strip = document.createElement("span");
        strip.className = "nf-strip";
        strip.style.display = "flex";
        strip.style.flexDirection = "column";
        strip.style.height = "auto";
        strip.style.lineHeight = "1.25em";
        strip.style.fontVariantNumeric = "tabular-nums";
        strip.style.willChange = isReduced ? "auto" : "transform";

        for (var d = 0; d <= 9; d += 1) {
            var cell = document.createElement("span");
            cell.className = "nf-digit";
            cell.textContent = String(d);
            cell.style.display = "flex";
            cell.style.alignItems = "center";
            cell.style.justifyContent = "center";
            cell.style.flex = "0 0 1.25em";
            cell.style.height = "1.25em";
            cell.style.lineHeight = "1.25em";
            strip.appendChild(cell);
        }

        var immediate = Boolean(isReduced || snapNow);
        strip.style.transition = "none";
        if (immediate) {
            strip.style.transform = "translateY(-" + (targetDigit * 10) + "%)";
        } else {
            strip.style.transform = "translateY(-" + (initialDigit * 10) + "%)";
        }

        slot.appendChild(strip);
        return {
            kind: "digit",
            slot: slot,
            strip: strip,
            startDigit: initialDigit,
            targetDigit: targetDigit,
            currentDigit: immediate ? targetDigit : initialDigit
        };
    };

    RollingCounter.prototype.setValue = function (numOrNull, opts) {
        if (this._destroyed) return;
        var o = opts || {};
        var decimals = o.decimals !== undefined ? clamp(toNum(o.decimals, this._decimals), 0, 6) : this._decimals;
        var unit = o.unit !== undefined ? String(o.unit) : this._unit;
        var fallback = o.fallback !== undefined ? String(o.fallback) : this._fallback;

        this._decimals = decimals;
        this._unit = unit;
        this._fallback = fallback;

        var isValid = numOrNull !== null &&
            numOrNull !== undefined &&
            numOrNull !== "" &&
            typeof numOrNull !== "boolean" &&
            Number.isFinite(Number(numOrNull));

        var numText = isValid ? Number(numOrNull).toFixed(decimals) : fallback;
        if (numText === this._lastNumText && isValid === this._lastValid && unit === this._lastUnit) {
            return;
        }

        var wasValid = this._lastValid;
        var nowMs = typeof performance !== "undefined" ? performance.now() : Date.now();
        var elapsedMs = (wasValid && this._lastUpdateMs > 0) ? (nowMs - this._lastUpdateMs) : 9999;
        this._lastUpdateMs = isValid ? nowMs : 0;
        var isReduced = this._motionTracker.isReduced();
        var shouldSnap = isReduced || elapsedMs < 380;
        var animDurationMs = shouldSnap ? 0 : 180;
        this._lastDurationMs = animDurationMs || 180;
        this._lastNumText = numText;
        this._lastValid = isValid;
        this._lastUnit = unit;

        var fullAria = isValid && unit ? (numText + " " + unit) : numText;
        this._el.setAttribute("aria-label", fullAria);
        this._el.setAttribute("data-value", numText);

        var transitionValue = shouldSnap
            ? "none"
            : ("transform " + animDurationMs + "ms cubic-bezier(0.22, 1, 0.36, 1)");

        var templateKey = "";
        if (!isValid) {
            templateKey = "FB|" + fallback;
        } else {
            var tplChars = [];
            for (var t = 0; t < numText.length; t += 1) {
                var tc = numText.charCodeAt(t);
                tplChars.push((tc >= 48 && tc <= 57) ? "#" : numText.charAt(t));
            }
            templateKey = "NUM|" + tplChars.join("") + "|" + unit;
        }

        // Fast path: template unchanged (same length and separator positions) -> reuse DOM nodes
        if (templateKey === this._templateKey && isValid && this._slots.length === numText.length) {
            var reuseDigits = [];
            var pendingAnim = [];
            for (var r = 0; r < numText.length; r += 1) {
                var rSlot = this._slots[r];
                var rCode = numText.charCodeAt(r);
                if (rSlot && rSlot.kind === "digit" && rCode >= 48 && rCode <= 57) {
                    var nextD = rCode - 48;
                    reuseDigits.push(nextD);
                    if (rSlot.currentDigit !== nextD) {
                        rSlot.targetDigit = nextD;
                        rSlot.currentDigit = nextD;
                        pendingAnim.push(rSlot);
                    }
                }
            }
            this._lastDigits = reuseDigits;
            if (shouldSnap) {
                if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
                    global.cancelAnimationFrame(this._rafId);
                    this._rafId = null;
                }
                for (var ps = 0; ps < this._slots.length; ps += 1) {
                    var snSlot = this._slots[ps];
                    if (snSlot && snSlot.kind === "digit") {
                        snSlot.strip.style.transition = "none";
                        snSlot.strip.style.transform = "translateY(-" + (snSlot.targetDigit * 10) + "%)";
                        snSlot.currentDigit = snSlot.targetDigit;
                    }
                }
            } else if (pendingAnim.length > 0) {
                if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
                    global.cancelAnimationFrame(this._rafId);
                    this._rafId = null;
                    var selfReuse = this;
                    this._rafId = global.requestAnimationFrame(function () {
                        selfReuse._rafId = null;
                        if (selfReuse._destroyed) return;
                        for (var p = 0; p < selfReuse._slots.length; p += 1) {
                            var sItem = selfReuse._slots[p];
                            if (sItem.kind === "digit") {
                                sItem.strip.style.transition = transitionValue;
                                sItem.strip.style.transform = "translateY(-" + (sItem.targetDigit * 10) + "%)";
                                sItem.currentDigit = sItem.targetDigit;
                            }
                        }
                    });
                } else {
                    for (var p2 = 0; p2 < pendingAnim.length; p2 += 1) {
                        var pSlot = pendingAnim[p2];
                        pSlot.strip.style.transition = transitionValue;
                        pSlot.strip.style.transform = "translateY(-" + (pSlot.targetDigit * 10) + "%)";
                    }
                }
            }
            return;
        }

        if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
            global.cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }

        while (this._el.firstChild) {
            this._el.removeChild(this._el.firstChild);
        }
        this._slots = [];
        this._unitEl = null;
        this._templateKey = templateKey;

        if (!isValid) {
            var fbSpan = document.createElement("span");
            fbSpan.className = "nf-char nf-fallback";
            fbSpan.setAttribute("aria-hidden", "true");
            fbSpan.textContent = fallback;
            fbSpan.style.display = "inline-block";
            fbSpan.style.height = "1.25em";
            fbSpan.style.lineHeight = "1.25em";
            fbSpan.style.verticalAlign = "baseline";
            this._el.appendChild(fbSpan);
            this._slots.push({ kind: "fallback", el: fbSpan });
            this._lastDigits = [];
            return;
        }

        var prevDigits = this._lastDigits;
        var newDigits = [];
        var digitSlots = [];

        // Count digit characters first to index right-to-left for smooth carry continuity
        var totalDigits = 0;
        for (var c = 0; c < numText.length; c += 1) {
            var ch = numText.charCodeAt(c);
            if (ch >= 48 && ch <= 57) totalDigits += 1;
        }

        var digitCursor = 0;
        var needsAnim = false;
        for (var i = 0; i < numText.length; i += 1) {
            var chStr = numText.charAt(i);
            var code = numText.charCodeAt(i);
            if (code >= 48 && code <= 57) {
                var digit = code - 48;
                var fromRight = totalDigits - 1 - digitCursor;
                var prevIdx = prevDigits.length - 1 - fromRight;
                var startDigit = (prevIdx >= 0 && prevIdx < prevDigits.length) ? prevDigits[prevIdx] : digit;
                var created = this._createDigitSlot(startDigit, digit, isReduced, shouldSnap);
                this._el.appendChild(created.slot);
                this._slots.push(created);
                digitSlots.push(created);
                if (startDigit !== digit) needsAnim = true;
                newDigits.push(digit);
                digitCursor += 1;
            } else {
                var sep = document.createElement("span");
                sep.className = "nf-char";
                sep.setAttribute("aria-hidden", "true");
                sep.textContent = chStr;
                sep.style.display = "inline-block";
                sep.style.height = "1.25em";
                sep.style.lineHeight = "1.25em";
                sep.style.verticalAlign = "baseline";
                this._el.appendChild(sep);
                this._slots.push({ kind: "char", el: sep, text: chStr });
            }
        }

        if (unit) {
            var unitSpan = document.createElement("span");
            unitSpan.className = "nf-unit";
            unitSpan.setAttribute("aria-hidden", "true");
            unitSpan.textContent = unit;
            unitSpan.style.display = "inline-block";
            unitSpan.style.height = "1.25em";
            unitSpan.style.lineHeight = "1.25em";
            unitSpan.style.verticalAlign = "baseline";
            this._el.appendChild(unitSpan);
            this._unitEl = unitSpan;
        }

        this._lastDigits = newDigits;

        if (!shouldSnap && needsAnim && digitSlots.length > 0 && typeof global.requestAnimationFrame === "function") {
            var self = this;
            this._rafId = global.requestAnimationFrame(function () {
                self._rafId = null;
                if (self._destroyed) return;
                for (var k = 0; k < digitSlots.length; k += 1) {
                    var item = digitSlots[k];
                    item.strip.style.transition = transitionValue;
                    item.strip.style.transform = "translateY(-" + (item.targetDigit * 10) + "%)";
                    item.currentDigit = item.targetDigit;
                }
            });
        } else {
            for (var k2 = 0; k2 < digitSlots.length; k2 += 1) {
                digitSlots[k2].strip.style.transition = "none";
                digitSlots[k2].strip.style.transform = "translateY(-" + (digitSlots[k2].targetDigit * 10) + "%)";
                digitSlots[k2].currentDigit = digitSlots[k2].targetDigit;
            }
        }
    };

    RollingCounter.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
            global.cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
        this._motionTracker.destroy();
    };

    /* ======================================================================
     * 2. AnimatedSignalPipeline (ported from Magic UI AnimatedBeam)
     * ====================================================================== */
    function AnimatedSignalPipeline(svgEl, options) {
        if (!(this instanceof AnimatedSignalPipeline)) {
            return new AnimatedSignalPipeline(svgEl, options);
        }
        if (!svgEl) {
            throw new Error("AnimatedSignalPipeline requires a valid <svg> element");
        }
        var opts = options || {};
        this._svg = svgEl;
        this._onPulse = typeof opts.onPulse === "function" ? opts.onPulse : null;
        this._paused = false;
        this._riskLevel = 0;
        this._config = {
            speedModel: "ctrv",
            conflictModel: "obb_ssm",
            stitching: true,
            cyclistVru: true
        };
        this._destroyed = false;
        this._rafId = null;
        this._lastFrameTs = null;
        this._phase = 0;
        this._lastCycleIdx = -1;

        this._probe = createColorProbe();
        this._palette = readThemePalette(this._probe);

        var self = this;
        this._motionTracker = createReducedMotionTracker(function () {
            self._syncAnimationLoop();
        });
        this._unobserveTheme = observeThemeChanges(function () {
            self._palette = readThemePalette(self._probe);
            self._applyColors();
        });

        this._ensureSvgStructure();
        this._measurePaths();

        attachResizeObserver(this._svg, this, function () {
            self._measurePaths();
        });

        this.setRiskLevel(0);
        this.setConfig(this._config);
        this._syncAnimationLoop();
    }

    AnimatedSignalPipeline.prototype._ensureSvgStructure = function () {
        var existingPaths = this._svg.querySelectorAll(".beam-path");
        var ns = "http://www.w3.org/2000/svg";

        if (existingPaths.length === 0) {
            // Synthesize clean responsive 5-node signal flow if host SVG is empty
            this._svg.setAttribute("viewBox", "0 0 920 120");
            var nodes = [
                { id: "detect", x: 92, label: "YOLOv8", sub: "Target Det" },
                { id: "track", x: 276, label: "ByteTrack+HSV", sub: "Appearance" },
                { id: "motion", x: 460, label: "Homography/CTRV", sub: "Kinematics" },
                { id: "conflict", x: 644, label: "OBB-SSM", sub: "TTC / PET" },
                { id: "alert", x: 828, label: "Dual-End Alert", sub: "Driver / VRU" }
            ];
            for (var s = 0; s < nodes.length - 1; s += 1) {
                var x1 = nodes[s].x + 68;
                var x2 = nodes[s + 1].x - 68;
                var dStr = "M " + x1 + " 60 L " + x2 + " 60";
                var basePath = document.createElementNS(ns, "path");
                basePath.setAttribute("class", "beam-path");
                basePath.setAttribute("data-segment", String(s));
                basePath.setAttribute("d", dStr);
                basePath.setAttribute("fill", "none");
                basePath.setAttribute("stroke-width", "2");
                this._svg.appendChild(basePath);

                var pulsePath = document.createElementNS(ns, "path");
                pulsePath.setAttribute("class", "beam-pulse");
                pulsePath.setAttribute("data-segment", String(s));
                pulsePath.setAttribute("d", dStr);
                pulsePath.setAttribute("fill", "none");
                pulsePath.setAttribute("stroke-width", "2.5");
                pulsePath.setAttribute("stroke-linecap", "round");
                this._svg.appendChild(pulsePath);
            }
            for (var n = 0; n < nodes.length; n += 1) {
                var g = document.createElementNS(ns, "g");
                g.setAttribute("data-stage", nodes[n].id);
                var rect = document.createElementNS(ns, "rect");
                rect.setAttribute("class", "node-box");
                rect.setAttribute("x", String(nodes[n].x - 68));
                rect.setAttribute("y", "32");
                rect.setAttribute("width", "136");
                rect.setAttribute("height", "56");
                rect.setAttribute("rx", "8");
                rect.setAttribute("fill", "var(--bg-raised, #232325)");
                rect.setAttribute("stroke", "var(--border-hairline, rgba(237,237,237,0.12))");
                g.appendChild(rect);

                var tMain = document.createElementNS(ns, "text");
                tMain.setAttribute("x", String(nodes[n].x));
                tMain.setAttribute("y", "56");
                tMain.setAttribute("text-anchor", "middle");
                tMain.setAttribute("fill", "var(--text-primary, #ededed)");
                tMain.setAttribute("font-size", "12");
                tMain.setAttribute("data-stage-label", nodes[n].id);
                tMain.textContent = nodes[n].label;
                g.appendChild(tMain);

                var tSub = document.createElementNS(ns, "text");
                tSub.setAttribute("x", String(nodes[n].x));
                tSub.setAttribute("y", "74");
                tSub.setAttribute("text-anchor", "middle");
                tSub.setAttribute("fill", "var(--text-muted, #85858b)");
                tSub.setAttribute("font-size", "10");
                tSub.setAttribute("data-stage-sub", nodes[n].id);
                tSub.textContent = nodes[n].sub;
                g.appendChild(tSub);

                this._svg.appendChild(g);
            }
        } else if (this._svg.querySelectorAll(".beam-pulse").length === 0) {
            // Auto-create overlay .beam-pulse paths matching existing .beam-path tracks
            for (var i = 0; i < existingPaths.length; i += 1) {
                var src = existingPaths[i];
                var clone = document.createElementNS(ns, "path");
                clone.setAttribute("class", "beam-pulse");
                clone.setAttribute("d", src.getAttribute("d") || "");
                clone.setAttribute("fill", "none");
                clone.setAttribute("stroke-linecap", "round");
                clone.setAttribute("stroke-width", src.getAttribute("stroke-width") || "2.5");
                if (src.hasAttribute("data-segment")) {
                    clone.setAttribute("data-segment", src.getAttribute("data-segment"));
                }
                if (src.parentNode) {
                    src.parentNode.insertBefore(clone, src.nextSibling);
                }
            }
        }

        this._basePaths = Array.prototype.slice.call(this._svg.querySelectorAll(".beam-path"));
        this._pulsePaths = Array.prototype.slice.call(this._svg.querySelectorAll(".beam-pulse"));
    };

    AnimatedSignalPipeline.prototype._measurePaths = function () {
        this._segments = [];
        for (var i = 0; i < this._pulsePaths.length; i += 1) {
            var p = this._pulsePaths[i];
            var len = 120;
            try {
                var measured = typeof p.getTotalLength === "function" ? p.getTotalLength() : 0;
                if (Number.isFinite(measured) && measured > 4) {
                    len = measured;
                }
            } catch (err) {
                len = 120;
            }
            var pulseLen = clamp(len * 0.30, 16, 46);
            this._segments.push({
                pulseEl: p,
                baseEl: this._basePaths[i] || null,
                length: len,
                pulseLen: pulseLen,
                index: i
            });
            p.style.fill = "none";
            p.style.strokeLinecap = "round";
        }
        this._applyStaticOrAnimatedPattern();
    };

    AnimatedSignalPipeline.prototype._applyColors = function () {
        var pal = this._palette;
        var alertColor = pal.neutralPipe;
        if (this._riskLevel === 3) alertColor = pal.critical;
        else if (this._riskLevel === 2 || this._riskLevel === 1) alertColor = pal.accent;

        var lastIdx = this._segments.length - 1;
        for (var i = 0; i < this._segments.length; i += 1) {
            var seg = this._segments[i];
            var isAlertSeg = (i === lastIdx) ||
                (seg.pulseEl.getAttribute("data-segment") === "alert") ||
                (seg.pulseEl.getAttribute("data-role") === "alert");
            var pulseColor = isAlertSeg ? alertColor : pal.accent;
            var baseColor = isAlertSeg && this._riskLevel >= 2 ? withAlpha(alertColor, 0.35) : pal.hairline;

            seg.pulseEl.style.stroke = pulseColor;
            if (seg.baseEl) {
                seg.baseEl.style.stroke = baseColor;
            }
        }

        // Update optional SVG alert indicator elements if present without overriding .is-selected CSS
        var alertBadge = this._svg.querySelector("[data-stage='alert'] rect.node-box, [data-stage='alert'] rect:not(.node-focus-ring), [data-node='alert'] rect:not(.node-focus-ring)");
        if (alertBadge) {
            alertBadge.style.stroke = this._riskLevel >= 2 ? alertColor : "";
        }
        this._svg.setAttribute("data-risk", String(this._riskLevel));
    };

    AnimatedSignalPipeline.prototype._applyStaticOrAnimatedPattern = function () {
        var isStatic = this._paused || this._motionTracker.isReduced();
        for (var i = 0; i < this._segments.length; i += 1) {
            var seg = this._segments[i];
            if (isStatic) {
                // Static crisp beam with subtle accent opacity and zero dash animation
                seg.pulseEl.style.strokeDasharray = "none";
                seg.pulseEl.style.strokeDashoffset = "0";
                seg.pulseEl.style.opacity = i === this._segments.length - 1
                    ? (this._riskLevel >= 2 ? "0.75" : "0.35")
                    : "0.32";
            } else {
                seg.pulseEl.style.strokeDasharray = seg.pulseLen.toFixed(1) + " " + (seg.length + seg.pulseLen).toFixed(1);
                seg.pulseEl.style.opacity = i === this._segments.length - 1 && this._riskLevel >= 2 ? "0.95" : "0.78";
            }
        }
        this._applyColors();
    };

    AnimatedSignalPipeline.prototype._syncAnimationLoop = function () {
        var shouldRun = !this._destroyed && !this._paused && !this._motionTracker.isReduced();
        this._applyStaticOrAnimatedPattern();

        if (!shouldRun) {
            if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
                global.cancelAnimationFrame(this._rafId);
                this._rafId = null;
            }
            this._lastFrameTs = null;
            return;
        }

        if (this._rafId !== null || typeof global.requestAnimationFrame !== "function") {
            return;
        }

        var self = this;
        function tick(now) {
            if (self._destroyed || self._paused || self._motionTracker.isReduced()) {
                self._rafId = null;
                return;
            }
            if (self._lastFrameTs === null) {
                self._lastFrameTs = now;
            }
            var dt = Math.min(0.08, Math.max(0, (now - self._lastFrameTs) / 1000));
            self._lastFrameTs = now;

            // Smooth 0.45 cycles/sec directional signal flow (well below 3 Hz threshold)
            var speedCps = self._riskLevel >= 3 ? 0.58 : 0.42;
            self._phase += dt * speedCps;

            var segCount = Math.max(1, self._segments.length);
            var activeIdx = Math.floor(self._phase * segCount) % segCount;
            if (activeIdx !== self._lastCycleIdx) {
                self._lastCycleIdx = activeIdx;
                if (self._onPulse) {
                    self._onPulse(activeIdx, self._riskLevel);
                }
            }

            for (var i = 0; i < self._segments.length; i += 1) {
                var seg = self._segments[i];
                var localT = (self._phase - i * 0.16) % 1;
                if (localT < 0) localT += 1;
                var totalTravel = seg.length + seg.pulseLen;
                var offset = seg.pulseLen - localT * totalTravel;
                seg.pulseEl.style.strokeDashoffset = offset.toFixed(2);
            }

            self._rafId = global.requestAnimationFrame(tick);
        }

        this._rafId = global.requestAnimationFrame(tick);
    };

    AnimatedSignalPipeline.prototype.setRiskLevel = function (level) {
        if (this._destroyed) return;
        this._riskLevel = parseRiskLevel(level);
        this._applyStaticOrAnimatedPattern();
    };

    AnimatedSignalPipeline.prototype.setConfig = function (cfg) {
        if (this._destroyed) return;
        var c = cfg || {};
        if (typeof c.speedModel === "string") this._config.speedModel = c.speedModel;
        if (typeof c.conflictModel === "string") this._config.conflictModel = c.conflictModel;
        if (typeof c.stitching === "boolean") this._config.stitching = c.stitching;
        if (typeof c.cyclistVru === "boolean") this._config.cyclistVru = c.cyclistVru;

        var trackLabel = this._svg.querySelector("[data-stage-label='track'], [data-node-title='track']");
        if (trackLabel) {
            trackLabel.textContent = this._config.stitching ? "ByteTrack+HSV" : "ByteTrack";
        }
        var motionLabel = this._svg.querySelector("[data-stage-label='motion'], [data-node-title='motion']");
        if (motionLabel) {
            var sm = this._config.speedModel.toLowerCase();
            motionLabel.textContent = sm === "kalman" ? "Homography/Kalman" : (sm === "ema" ? "Homography/EMA" : "Homography/CTRV");
        }
        var conflictLabel = this._svg.querySelector("[data-stage-label='conflict'], [data-node-title='conflict']");
        if (conflictLabel) {
            var cm = this._config.conflictModel.toLowerCase();
            conflictLabel.textContent = cm === "cvm_circle" ? "CVM-Circle" : "OBB-SSM";
        }
        var conflictSub = this._svg.querySelector("[data-stage-sub='conflict'], [data-node-sub='conflict']");
        if (conflictSub) {
            conflictSub.textContent = this._config.cyclistVru ? "TTC / PET + VRU" : "TTC / PET";
        }
    };

    AnimatedSignalPipeline.prototype.setPaused = function (paused) {
        if (this._destroyed) return;
        this._paused = Boolean(paused);
        this._syncAnimationLoop();
    };

    AnimatedSignalPipeline.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
            global.cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
        detachResizeObserver(this);
        this._unobserveTheme();
        this._motionTracker.destroy();
        if (this._probe && this._probe.parentNode) {
            this._probe.parentNode.removeChild(this._probe);
        }
    };

    /* ======================================================================
     * 3. CampusRoadTwinCanvas (Bird's-Eye Metric Road Twin + Physics Engine)
     * ====================================================================== */
    var DEFAULT_BOUNDS = { xMin: -3.2, xMax: 3.2, yMin: 0.0, yMax: 15.0 };

    function normalizeBounds(b) {
        if (!b) return { xMin: DEFAULT_BOUNDS.xMin, xMax: DEFAULT_BOUNDS.xMax, yMin: DEFAULT_BOUNDS.yMin, yMax: DEFAULT_BOUNDS.yMax };
        if (Array.isArray(b) && b.length >= 4) {
            return {
                xMin: toNum(b[0], DEFAULT_BOUNDS.xMin),
                xMax: toNum(b[1], DEFAULT_BOUNDS.xMax),
                yMin: toNum(b[2], DEFAULT_BOUNDS.yMin),
                yMax: toNum(b[3], DEFAULT_BOUNDS.yMax)
            };
        }
        return {
            xMin: toNum(b.xMin !== undefined ? b.xMin : b.x_min, DEFAULT_BOUNDS.xMin),
            xMax: toNum(b.xMax !== undefined ? b.xMax : b.x_max, DEFAULT_BOUNDS.xMax),
            yMin: toNum(b.yMin !== undefined ? b.yMin : b.y_min, DEFAULT_BOUNDS.yMin),
            yMax: toNum(b.yMax !== undefined ? b.yMax : b.y_max, DEFAULT_BOUNDS.yMax)
        };
    }

    function computeCvmCircleTtc(veh, vru, horizonS) {
        var horizon = horizonS || 3.5;
        var dx = vru.x - veh.x;
        var dy = vru.y - veh.y;
        var rvx = (vru.vx || 0) - (veh.vx || 0);
        var rvy = (vru.vy || 0) - (veh.vy || 0);
        var currentDist = Math.hypot(dx, dy);
        var halfL = (veh.lengthM || 4.5) * 0.5;
        var halfW = (veh.widthM || 1.9) * 0.5;
        var vehRadius = Math.max(halfW * 0.85, Math.min(2.8, halfL * 0.55));
        var pedRadius = vru.radiusM || 0.42;
        var safeSum = vehRadius + pedRadius;

        var vRelSq = rvx * rvx + rvy * rvy;
        var tCpa = vRelSq > 1e-6 ? clamp(-(dx * rvx + dy * rvy) / vRelSq, 0, horizon) : 0;
        var cx = dx + rvx * tCpa;
        var cy = dy + rvy * tCpa;
        var cpaCenterDist = Math.hypot(cx, cy);
        var minClearance = Math.max(0, cpaCenterDist - safeSum);

        var ttc = null;
        if (currentDist <= safeSum) {
            ttc = 0;
        } else if (vRelSq > 1e-6) {
            var b = dx * rvx + dy * rvy;
            var c = dx * dx + dy * dy - safeSum * safeSum;
            var disc = b * b - vRelSq * c;
            if (disc >= 0) {
                var tEnter = (-b - Math.sqrt(disc)) / vRelSq;
                if (tEnter >= 0 && tEnter <= horizon) {
                    ttc = tEnter;
                }
            }
        }

        var tContact = ttc !== null ? ttc : tCpa;
        var midX = (veh.x + (veh.vx || 0) * tContact + vru.x + (vru.vx || 0) * tContact) * 0.5;
        var midY = (veh.y + (veh.vy || 0) * tContact + vru.y + (vru.vy || 0) * tContact) * 0.5;
        return {
            currentDist: currentDist,
            minDistanceM: minClearance,
            ttcSeconds: ttc,
            conflictGroundXY: [midX, midY]
        };
    }

    function resolveHeadingRad(val, fallback) {
        return isFiniteNum(val) ? val : (fallback !== undefined ? fallback : Math.PI * 0.5);
    }

    function rolloutVehicleTrajectory(veh, horizonS, steps, speedModel) {
        var count = steps || 35;
        var totalHorizon = horizonS || 3.5;
        var dt = totalHorizon / count;
        var model = String(speedModel || "ctrv").toLowerCase();
        var x = veh.x;
        var y = veh.y;
        var psi = resolveHeadingRad(veh.headingRad, Math.PI * 0.5);
        var v = Math.hypot(veh.vx || 0, veh.vy || 0);
        var a = veh.accelMs2 || 0;
        var omega = veh.yawRate || 0;

        if (model === "ema") {
            v = v * 0.88;
            a = 0;
            omega = 0;
        } else if (model === "kalman") {
            omega = 0;
        }

        var pts = [{ x: x, y: y, psi: psi, v: v, a: a, t: 0 }];
        for (var i = 1; i <= count; i += 1) {
            var nextV = Math.max(0, v + a * dt);
            var avgV = 0.5 * (v + nextV);
            if (Math.abs(omega) > 0.01) {
                var nextPsi = psi + omega * dt;
                x += (avgV / omega) * (Math.sin(nextPsi) - Math.sin(psi));
                y += (avgV / omega) * (-Math.cos(nextPsi) + Math.cos(psi));
                psi = nextPsi;
            } else {
                x += avgV * Math.cos(psi) * dt;
                y += avgV * Math.sin(psi) * dt;
            }
            v = nextV;
            pts.push({ x: x, y: y, psi: psi, v: v, a: a, t: i * dt });
        }
        return pts;
    }

    function computeCpaAndTtc(veh, vru, horizonS, speedModel) {
        var totalHorizon = horizonS || 3.5;
        var steps = Math.max(1, Math.round(totalHorizon / 0.05));
        var traj = rolloutVehicleTrajectory(veh, totalHorizon, steps, speedModel);
        var x0 = traj[0].x;
        var y0 = traj[0].y;
        var psi0 = traj[0].psi;
        var v0 = traj[0].v;
        var a0 = traj[0].a;
        var px = vru.x;
        var py = vru.y;
        var pvx = vru.vx;
        var pvy = vru.vy;

        var halfL = (veh.lengthM || 4.5) * 0.5;
        var halfW = (veh.widthM || 1.9) * 0.5;
        var safeR = vru.radiusM || 0.42;

        var currentDist = Math.hypot(px - x0, py - y0);
        var minClearance = Infinity;
        var bestX = x0;
        var bestY = y0;
        var ttc = null;

        // If vehicle is effectively stationary and not overlapping right now, no TTC conflict
        if (v0 < 0.25 && a0 <= 0.1) {
            var dx0 = px - x0;
            var dy0 = py - y0;
            var c0 = Math.cos(psi0);
            var s0 = Math.sin(psi0);
            var lx0 = dx0 * c0 + dy0 * s0;
            var ly0 = -dx0 * s0 + dy0 * c0;
            var cx0 = clamp(lx0, -halfL, halfL);
            var cy0 = clamp(ly0, -halfW, halfW);
            var clear0 = Math.hypot(lx0 - cx0, ly0 - cy0) - safeR;
            return {
                currentDist: currentDist,
                minDistanceM: Math.max(0, clear0),
                ttcSeconds: clear0 <= 0.05 ? 0 : null,
                conflictGroundXY: [x0 + cx0 * c0 - cy0 * s0, y0 + cx0 * s0 + cy0 * c0]
            };
        }

        for (var i = 0; i < traj.length; i += 1) {
            var st = traj[i];
            var t = st.t;
            var dx = (px + pvx * t) - st.x;
            var dy = (py + pvy * t) - st.y;
            var cosP = Math.cos(st.psi);
            var sinP = Math.sin(st.psi);
            var lx = dx * cosP + dy * sinP;
            var ly = -dx * sinP + dy * cosP;
            var cx = clamp(lx, -halfL, halfL);
            var cy = clamp(ly, -halfW, halfW);
            var clearance = Math.hypot(lx - cx, ly - cy) - safeR;

            if (clearance < minClearance) {
                minClearance = clearance;
                bestX = st.x + (cx * cosP - cy * sinP + dx) * 0.5;
                bestY = st.y + (cx * sinP + cy * cosP + dy) * 0.5;
            }
            if (ttc === null && clearance <= 0.12) {
                ttc = t;
            }
        }

        return {
            currentDist: currentDist,
            minDistanceM: Math.max(0, minClearance),
            ttcSeconds: ttc,
            conflictGroundXY: [bestX, bestY]
        };
    }

    function predictCtrvArc(veh, horizonS, steps, speedModel) {
        return rolloutVehicleTrajectory(veh, horizonS || 2.5, steps || 15, speedModel);
    }

    var SCENARIO_CYCLE_S = 11.2;

    function isCyclistVruKind(item) {
        if (!item) return false;
        var k = String(item.category || item.kind || "").toLowerCase();
        return k === "bicycle" || k === "motorcycle";
    }

    function integrateCtrvState(tSec) {
        var psiTarget = Math.PI * 0.5;
        var v0 = 5.4;
        var omega0 = 0.28;
        var tC = 0.18;
        var a1 = -7.0;
        var t1 = 0.50;
        var a2 = -0.40;
        var t2 = 1.25;
        var a3 = -3.4;
        var v1 = v0 + a1 * t1;
        var v2 = v1 + a2 * t2;
        var t1End = tC + t1;
        var t2End = t1End + t2;
        var tStop = t2End + v2 / (-a3);
        var tGo = tStop + 2.55;
        var accelDep = 3.8;

        function phaseParams(tau, curPsi) {
            if (tau < tC) {
                return { v: v0, a: 0.0, omega: omega0 };
            }
            if (tau < t1End) {
                var vA = v0 + a1 * (tau - tC);
                return { v: vA, a: a1, omega: omega0 * (vA / v0) };
            }
            if (tau < t2End) {
                var vB = v1 + a2 * (tau - t1End);
                return { v: vB, a: a2, omega: omega0 * (vB / v0) };
            }
            if (tau < tStop) {
                var vC = Math.max(0.0, v2 + a3 * (tau - t2End));
                return { v: vC, a: a3, omega: omega0 * (vC / v0) };
            }
            if (tau < tGo) {
                return { v: 0.0, a: 0.0, omega: 0.0 };
            }
            var u = tau - tGo;
            var vDep = Math.min(7.8, accelDep * u);
            var aDep = vDep < 7.8 ? accelDep : 0.0;
            var err = curPsi - psiTarget;
            var omegaDep = (vDep > 0.05 && Math.abs(err) > 0.002)
                ? -clamp(err * 1.4, -0.45, 0.45) * Math.min(1.0, vDep / 1.5)
                : 0.0;
            return { v: vDep, a: aDep, omega: omegaDep };
        }

        var x = 1.40;
        var y = 0.60;
        var psi = Math.PI * 0.5;
        var steps = Math.max(1, Math.ceil(tSec / 0.02));
        var dtStep = tSec > 0 ? (tSec / steps) : 0;
        for (var i = 0; i < steps; i += 1) {
            var tau = i * dtStep;
            var p1 = phaseParams(tau, psi);
            var psiMid = psi + p1.omega * (0.5 * dtStep);
            var pMid = phaseParams(tau + 0.5 * dtStep, psiMid);
            x += pMid.v * Math.cos(psiMid) * dtStep;
            y += pMid.v * Math.sin(psiMid) * dtStep;
            psi += pMid.omega * dtStep;
            if (tau >= tGo && Math.abs(psi - psiTarget) < 0.005) {
                psi = psiTarget;
            }
        }
        var end = phaseParams(tSec, psi);
        return { x: x, y: y, psi: psi, v: end.v, a: end.a, omega: end.omega };
    }

    function buildScenarioState(scenarioId, cycleTimeS, config) {
        var t = ((cycleTimeS % SCENARIO_CYCLE_S) + SCENARIO_CYCLE_S) % SCENARIO_CYCLE_S;
        var vehicles = [];
        var vrus = [];
        var parkingZones = [
            [[1.85, 2.2], [3.05, 2.2], [3.05, 6.8], [1.85, 6.8]]
        ];
        var stitchingOn = !(config && config.stitching === false);
        var exitHideY = 17.6; // Far boundary beyond calibrated 15.0m + 2.6m apron

        if (scenarioId === "cyclist_vru_cutin") {
            // Scenario 2: Cyclist blind-spot cut-in past parked car V-02, departing clean past far boundary before wrap
            var s2Tc = 0.22;
            var s2V0 = 5.15;
            var s2A1 = -6.5;
            var s2T1 = 0.50;
            var s2A2 = -0.45;
            var s2T2 = 1.25;
            var s2A3 = -3.4;
            var s2Y0 = 0.58;
            var s2Y1 = s2Y0 + s2V0 * s2Tc;
            var s2V1 = s2V0 + s2A1 * s2T1;
            var s2Y2 = s2Y1 + s2V0 * s2T1 + 0.5 * s2A1 * s2T1 * s2T1;
            var s2V2 = s2V1 + s2A2 * s2T2;
            var s2Y3 = s2Y2 + s2V1 * s2T2 + 0.5 * s2A2 * s2T2 * s2T2;
            var s2TStop = s2Tc + s2T1 + s2T2 + s2V2 / (-s2A3);
            var s2TGo = s2TStop + 2.60;
            var s2YStop = s2Y3 + 0.5 * s2V2 * s2V2 / (-s2A3);
            var s2AccelDep = 3.8;

            var suvY = s2Y0;
            var suvVy = s2V0;
            var suvAy = 0.0;
            if (t < s2Tc) {
                suvVy = s2V0;
                suvAy = 0.0;
                suvY = s2Y0 + s2V0 * t;
            } else if (t < s2Tc + s2T1) {
                var su1 = t - s2Tc;
                suvAy = s2A1;
                suvVy = s2V0 + s2A1 * su1;
                suvY = s2Y1 + s2V0 * su1 + 0.5 * s2A1 * su1 * su1;
            } else if (t < s2Tc + s2T1 + s2T2) {
                var su2 = t - s2Tc - s2T1;
                suvAy = s2A2;
                suvVy = s2V1 + s2A2 * su2;
                suvY = s2Y2 + s2V1 * su2 + 0.5 * s2A2 * su2 * su2;
            } else if (t < s2TStop) {
                var su3 = t - s2Tc - s2T1 - s2T2;
                suvAy = s2A3;
                suvVy = Math.max(0.0, s2V2 + s2A3 * su3);
                suvY = s2Y3 + s2V2 * su3 + 0.5 * s2A3 * su3 * su3;
            } else if (t < s2TGo) {
                suvVy = 0.0;
                suvAy = 0.0;
                suvY = s2YStop;
            } else {
                var su4 = t - s2TGo;
                var s2Ta = 2.40;
                if (su4 < s2Ta) {
                    suvAy = s2AccelDep;
                    suvVy = suvAy * su4;
                    suvY = s2YStop + 0.5 * suvAy * su4 * su4;
                } else {
                    suvAy = 0.0;
                    suvVy = s2AccelDep * s2Ta;
                    suvY = s2YStop + 0.5 * s2AccelDep * s2Ta * s2Ta + suvVy * (su4 - s2Ta);
                }
            }

            if (suvY - 2.35 <= exitHideY) {
                vehicles.push({
                    trackId: 1,
                    kind: "suv",
                    x: -0.32,
                    y: suvY,
                    vx: 0,
                    vy: suvVy,
                    ax: 0,
                    ay: suvAy,
                    accelMs2: suvAy,
                    headingRad: Math.PI * 0.5,
                    yawRate: 0,
                    lengthM: 4.7,
                    widthM: 1.95,
                    speedKmh: suvVy * 3.6,
                    inParkingZone: false
                });
            }

            vehicles.push({
                trackId: 2,
                kind: "car",
                x: 2.40,
                y: 4.60,
                vx: 0,
                vy: 0,
                ax: 0,
                ay: 0,
                accelMs2: 0,
                headingRad: Math.PI * 0.5,
                yawRate: 0,
                lengthM: 4.2,
                widthM: 1.85,
                speedKmh: 0,
                inParkingZone: true
            });

            // Cyclist B-12 emerges past parked car V-02 laterally across SUV lane and exits cleanly past far horizon
            var bX0 = 2.85;
            var bY0 = 9.05;
            var bVx1 = -0.45;
            var bTTurn = 1.35;
            var bVx2 = -1.45;
            var cycX, cycY, cycVx, cycVy = 0.14;
            if (t < bTTurn) {
                cycVx = bVx1;
                cycX = bX0 + cycVx * t;
                cycY = bY0 + cycVy * t;
            } else {
                var bxTurn = bX0 + bVx1 * bTTurn;
                var byTurn = bY0 + cycVy * bTTurn;
                cycVx = bVx2;
                cycX = bxTurn + cycVx * (t - bTTurn);
                cycY = byTurn + cycVy * (t - bTTurn);
                if (cycX < -2.65) {
                    var dtDone = (-2.65 - bxTurn) / bVx2;
                    cycX = -2.65;
                    cycVx = 0.0;
                    cycVy = 1.65;
                    cycY = byTurn + 0.14 * dtDone + cycVy * (t - bTTurn - dtDone);
                }
            }
            var cycTrackId = (!stitchingOn && t >= 1.15) ? 19 : 12;

            if (cycY - 0.9 <= exitHideY) {
                vrus.push({
                    trackId: cycTrackId,
                    kind: "bicycle",
                    category: "bicycle",
                    x: cycX,
                    y: cycY,
                    vx: cycVx,
                    vy: cycVy,
                    radiusM: 0.50,
                    lengthM: 1.8,
                    widthM: 0.7,
                    headingRad: Math.atan2(cycVy, cycVx)
                });
            }

            // Periodic over SCENARIO_CYCLE_S so sidewalk pedestrian never snaps at cycle wrap
            var ped5Omega = (2 * Math.PI) / SCENARIO_CYCLE_S;
            vrus.push({
                trackId: 5,
                kind: "pedestrian",
                category: "pedestrian",
                x: -2.75,
                y: 11.5 + 0.35 * Math.sin(t * ped5Omega),
                vx: 0,
                vy: 0.35 * ped5Omega * Math.cos(t * ped5Omega),
                radiusM: 0.40
            });
        } else if (scenarioId === "ctrv_curve_yield") {
            // Scenario 3: S-curve CTRV yield trajectory with lane counter-steer and off-screen departure
            parkingZones = [];
            var ctrv = integrateCtrvState(t);

            if (ctrv.y - 2.25 <= exitHideY) {
                vehicles.push({
                    trackId: 1,
                    kind: "car",
                    x: ctrv.x,
                    y: ctrv.y,
                    vx: Math.cos(ctrv.psi) * ctrv.v,
                    vy: Math.sin(ctrv.psi) * ctrv.v,
                    ax: Math.cos(ctrv.psi) * ctrv.a,
                    ay: Math.sin(ctrv.psi) * ctrv.a,
                    accelMs2: ctrv.a,
                    headingRad: ctrv.psi,
                    yawRate: ctrv.omega,
                    lengthM: 4.5,
                    widthM: 1.9,
                    speedKmh: ctrv.v * 3.6,
                    inParkingZone: false
                });
            }

            // Pedestrian P-04 crosses zebra crosswalk and exits off left sidewalk beyond t=8.2s
            var pedX, pedVx;
            if (t <= 1.45) {
                pedVx = -0.44;
                pedX = 2.55 + pedVx * t;
            } else {
                pedVx = -1.35;
                pedX = (2.55 - 0.44 * 1.45) + pedVx * (t - 1.45);
            }
            if (pedX <= -2.90) {
                pedX = -2.90;
                pedVx = 0.0;
            }
            if (t <= 8.2) {
                vrus.push({
                    trackId: 4,
                    kind: "pedestrian",
                    category: "pedestrian",
                    x: pedX,
                    y: 9.40,
                    vx: pedVx,
                    vy: 0,
                    radiusM: 0.42
                });
            }
        } else {
            // Scenario 1 (default): Crosswalk two-stage braking (L2 early caution -> L3 critical brake -> yield -> depart)
            var s1Tc = 0.15;
            var s1V0 = 2.95;
            var s1A1 = -0.30;
            var s1T1 = 0.70;
            var s1A2 = -0.25;
            var s1T2 = 0.70;
            var s1A3 = -3.2;
            var s1Y0 = 0.50;
            var s1Y1 = s1Y0 + s1V0 * s1Tc;
            var s1V1 = s1V0 + s1A1 * s1T1;
            var s1Y2 = s1Y1 + s1V0 * s1T1 + 0.5 * s1A1 * s1T1 * s1T1;
            var s1V2 = s1V1 + s1A2 * s1T2;
            var s1Y3 = s1Y2 + s1V1 * s1T2 + 0.5 * s1A2 * s1T2 * s1T2;
            var s1YStop = s1Y3 + 0.5 * s1V2 * s1V2 / (-s1A3);
            var s1TStop = s1Tc + s1T1 + s1T2 + s1V2 / (-s1A3);
            var s1TGo = s1TStop + 2.35;
            var s1AccelDep = 3.6;

            var vY = s1Y0;
            var vSpeedMs = s1V0;
            var vAccel = 0.0;

            if (t < s1Tc) {
                vSpeedMs = s1V0;
                vAccel = 0.0;
                vY = s1Y0 + s1V0 * t;
            } else if (t < s1Tc + s1T1) {
                var b1 = t - s1Tc;
                vAccel = s1A1;
                vSpeedMs = s1V0 + s1A1 * b1;
                vY = s1Y1 + s1V0 * b1 + 0.5 * s1A1 * b1 * b1;
            } else if (t < s1Tc + s1T1 + s1T2) {
                var b2 = t - s1Tc - s1T1;
                vAccel = s1A2;
                vSpeedMs = s1V1 + s1A2 * b2;
                vY = s1Y2 + s1V1 * b2 + 0.5 * s1A2 * b2 * b2;
            } else if (t < s1TStop) {
                var b3 = t - s1Tc - s1T1 - s1T2;
                vAccel = s1A3;
                vSpeedMs = Math.max(0.0, s1V2 + s1A3 * b3);
                vY = s1Y3 + s1V2 * b3 + 0.5 * s1A3 * b3 * b3;
            } else if (t < s1TGo) {
                vY = s1YStop;
                vSpeedMs = 0.0;
                vAccel = 0.0;
            } else {
                var b4 = t - s1TGo;
                var s1Ta = 2.40;
                if (b4 < s1Ta) {
                    vAccel = s1AccelDep;
                    vSpeedMs = vAccel * b4;
                    vY = s1YStop + 0.5 * vAccel * b4 * b4;
                } else {
                    vAccel = 0.0;
                    vSpeedMs = s1AccelDep * s1Ta;
                    vY = s1YStop + 0.5 * s1AccelDep * s1Ta * s1Ta + vSpeedMs * (b4 - s1Ta);
                }
            }

            if (vY - 2.25 <= exitHideY) {
                vehicles.push({
                    trackId: 1,
                    kind: "car",
                    x: -0.65,
                    y: vY,
                    vx: 0,
                    vy: vSpeedMs,
                    ax: 0,
                    ay: vAccel,
                    accelMs2: vAccel,
                    headingRad: Math.PI * 0.5,
                    yawRate: 0,
                    lengthM: 4.5,
                    widthM: 1.9,
                    speedKmh: vSpeedMs * 3.6,
                    inParkingZone: false
                });
            }

            // Pedestrian P-07 clears zebra crosswalk cleanly onto left sidewalk before departure
            var p7x, p7vx;
            if (t < 1.90) {
                p7vx = -1.30;
                p7x = 2.80 + p7vx * t;
            } else {
                p7vx = -1.90;
                p7x = (2.80 - 1.30 * 1.90) + p7vx * (t - 1.90);
            }
            if (p7x <= -2.90) {
                p7x = -2.90;
                p7vx = 0.0;
            }
            if (t <= 8.2) {
                vrus.push({
                    trackId: 7,
                    kind: "pedestrian",
                    category: "pedestrian",
                    x: p7x,
                    y: 9.40,
                    vx: p7vx,
                    vy: 0,
                    radiusM: 0.42
                });
            }

            // Periodic over SCENARIO_CYCLE_S so sidewalk pedestrian P-03 has smooth closed loop
            var p3Omega = (2 * Math.PI) / SCENARIO_CYCLE_S;
            vrus.push({
                trackId: 3,
                kind: "pedestrian",
                category: "pedestrian",
                x: 2.75,
                y: 11.8 - 0.40 * Math.cos(t * p3Omega),
                vx: 0,
                vy: 0.40 * p3Omega * Math.sin(t * p3Omega),
                radiusM: 0.40
            });
        }

        return {
            bounds: { xMin: -3.2, xMax: 3.2, yMin: 0.0, yMax: 15.0 },
            parkingZones: parkingZones,
            vehicles: vehicles,
            vrus: vrus,
            cycleFade: t < 0.25 ? (t / 0.25) : 1.0
        };
    }

    function CampusRoadTwinCanvas(canvasEl, options) {
        if (!(this instanceof CampusRoadTwinCanvas)) {
            return new CampusRoadTwinCanvas(canvasEl, options);
        }
        if (!canvasEl) {
            throw new Error("CampusRoadTwinCanvas requires a valid <canvas> element");
        }
        var opts = options || {};
        this._canvas = canvasEl;
        this._ctx = canvasEl.getContext("2d");
        this._onTelemetry = typeof opts.onTelemetry === "function" ? opts.onTelemetry : null;
        this._onRenderState = typeof opts.onRenderState === "function" ? opts.onRenderState : null;
        this._speedLimitKmh = toNum(opts.speedLimitKmh, 30.0);
        this._config = {
            speedModel: (opts.config && opts.config.speedModel) || "ctrv",
            conflictModel: (opts.config && opts.config.conflictModel) || "obb_ssm",
            stitching: opts.config && opts.config.stitching !== undefined ? Boolean(opts.config.stitching) : true,
            cyclistVru: opts.config && opts.config.cyclistVru !== undefined ? Boolean(opts.config.cyclistVru) : true
        };

        this._mode = "simulation";
        this._scenarioId = "crosswalk_braking";
        this._userPaused = false;
        this._offscreenPaused = false;
        this._destroyed = false;

        this._lastDrawTs = null;
        this._lastTelemetryTs = 0;
        this._rafId = null;
        this._bgTimerId = null;
        this._lastBgTs = null;

        // 0.5m PET spatial occupancy grid & swept centerline vehicle trails
        this._petCells = Object.create(null);
        this._lastPetTimeS = null;
        this._vehTrails = Object.create(null);
        this._replayPrevTracks = Object.create(null);
        this._lastReplayFrameIdx = -1;

        this._cssWidth = 0;
        this._cssHeight = 0;
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);

        this._probe = createColorProbe();
        this._palette = readThemePalette(this._probe);

        var self = this;
        this._motionTracker = createReducedMotionTracker(function (isReduced) {
            if (isReduced) {
                self._stopLoop();
                self._renderCurrent();
                self._emitSimTelemetry(true);
            } else {
                self._syncLoop();
            }
        });
        this._simTimeS = (this._userPaused || this._motionTracker.isReduced()) ? 1.55 : 0.35;
        this._simFrameCount = Math.floor(this._simTimeS * 25 + 1e-9) + 1;
        this._lastEmittedRisk = -1;
        this._lastEmittedAlerts = -1;
        this._unobserveTheme = observeThemeChanges(function () {
            self.refreshTheme();
        });

        this._replayState = {
            frameIndex: 0,
            timestampS: 0,
            bounds: normalizeBounds(null),
            vehicles: [],
            vrus: [],
            conflicts: [],
            parkingZones: [],
            label: ""
        };

        this._currentRenderState = {
            bounds: normalizeBounds(null),
            parkingZones: [],
            vehicles: [],
            vrus: [],
            conflicts: [],
            label: "",
            simTimeS: 0,
            maxRisk: 0
        };

        attachResizeObserver(this._canvas, this, function () {
            self._measureCanvas();
            self._renderCurrent();
        });

        if (typeof IntersectionObserver === "function") {
            this._intersectionObserver = new IntersectionObserver(function (entries) {
                var visible = entries && entries[0] ? entries[0].isIntersecting : true;
                var wasOffscreen = self._offscreenPaused;
                self._offscreenPaused = !visible;
                if (visible && wasOffscreen) {
                    self._measureCanvas();
                    self._renderCurrent();
                }
                self._syncLoop();
            }, { threshold: 0.05 });
            this._intersectionObserver.observe(this._canvas);
        } else {
            this._intersectionObserver = null;
        }

        this._measureCanvas();
        this._seedInitialPetTrail();
        this._stepSimulation(0, true);
        this._syncLoop();
    }

    CampusRoadTwinCanvas.prototype._measureCanvas = function () {
        var rect = this._canvas.getBoundingClientRect();
        var w = Math.max(0, rect.width || this._canvas.clientWidth || 0);
        var h = Math.max(0, rect.height || this._canvas.clientHeight || 0);
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        this._cssWidth = w;
        this._cssHeight = h;
        if (w >= 2 && h >= 2) {
            var pw = Math.round(w * this._dpr);
            var ph = Math.round(h * this._dpr);
            if (this._canvas.width !== pw || this._canvas.height !== ph) {
                this._canvas.width = pw;
                this._canvas.height = ph;
            }
        }
    };

    CampusRoadTwinCanvas.prototype._seedInitialPetTrail = function () {
        this._petCells = Object.create(null);
        this._vehTrails = Object.create(null);
        var useCvm = this._config && this._config.conflictModel === "cvm_circle";
        var cycleTime = ((this._simTimeS % SCENARIO_CYCLE_S) + SCENARIO_CYCLE_S) % SCENARIO_CYCLE_S;
        var startT = Math.max(0, cycleTime - 2.0);
        var activeCyclistVru = !(this._config && this._config.cyclistVru === false);
        for (var t = startT; t <= cycleTime + 1e-6; t += 0.20) {
            var st = buildScenarioState(this._scenarioId, t, this._config);
            this._recordVehTrails(st.vehicles, t);
            if (!useCvm) {
                var activeVrus = activeCyclistVru
                    ? st.vrus
                    : st.vrus.filter(function (v) { return !isCyclistVruKind(v); });
                this._recordPetGrid(st.vehicles, activeVrus, t);
            }
        }
    };

    CampusRoadTwinCanvas.prototype._recordVehTrails = function (vehicles, timeS) {
        for (var vi = 0; vi < vehicles.length; vi += 1) {
            var v = vehicles[vi];
            if (v.inParkingZone) continue;
            var spdMs = Math.hypot(v.vx || 0, v.vy || 0);
            if (spdMs < 0.35 && (v.speedKmh === undefined || v.speedKmh < 1.2)) continue;
            var key = String(v.trackId);
            var trail = this._vehTrails[key];
            if (!trail) {
                trail = [];
                this._vehTrails[key] = trail;
            }
            var last = trail.length > 0 ? trail[trail.length - 1] : null;
            if (!last || Math.hypot(v.x - last.x, v.y - last.y) >= 0.08 || Math.abs(timeS - last.t) >= 0.08) {
                trail.push({ x: v.x, y: v.y, t: timeS });
            }
            while (trail.length > 32 || (trail.length > 0 && timeS - trail[0].t > 2.4)) {
                trail.shift();
            }
        }
        var tKeys = Object.keys(this._vehTrails);
        for (var ti = 0; ti < tKeys.length; ti += 1) {
            var arr = this._vehTrails[tKeys[ti]];
            if (!arr || arr.length === 0 || (timeS - arr[arr.length - 1].t > 2.4)) {
                delete this._vehTrails[tKeys[ti]];
            }
        }
    };

    CampusRoadTwinCanvas.prototype._recordPetGrid = function (vehicles, vrus, timeS) {
        var cellSize = 0.5;
        if (this._lastPetTimeS !== null && timeS < this._lastPetTimeS - 1e-6) {
            this._petCells = Object.create(null);
            this._lastPetTimeS = null;
        }
        var prevTimeS = this._lastPetTimeS !== null ? this._lastPetTimeS : timeS;

        var curVehMap = Object.create(null);
        for (var vi = 0; vi < vehicles.length; vi += 1) {
            var v = vehicles[vi];
            if (v.inParkingZone || (v.speedKmh !== undefined && v.speedKmh < 0.4)) continue;
            var halfL = (v.lengthM || 4.5) * 0.5;
            var halfW = (v.widthM || 1.9) * 0.5;
            var hRad = resolveHeadingRad(v.headingRad, Math.PI * 0.5);
            var cosP = Math.cos(hRad);
            var sinP = Math.sin(hRad);

            for (var lx = -halfL; lx <= halfL + 0.01; lx += cellSize) {
                for (var ly = -halfW; ly <= halfW + 0.01; ly += cellSize) {
                    var gx = v.x + lx * cosP - ly * sinP;
                    var gy = v.y + lx * sinP + ly * cosP;
                    var cx = Math.floor(gx / cellSize);
                    var cy = Math.floor(gy / cellSize);
                    var key = cx + "," + cy;
                    curVehMap[key] = { cx: cx, cy: cy, id: v.trackId };
                }
            }
        }

        var curVruMap = Object.create(null);
        for (var pi = 0; pi < vrus.length; pi += 1) {
            var p = vrus[pi];
            var pcx = Math.floor(p.x / cellSize);
            var pcy = Math.floor(p.y / cellSize);
            var pkey = pcx + "," + pcy;
            curVruMap[pkey] = { cx: pcx, cy: pcy, id: p.trackId };
        }

        var existingKeys = Object.keys(this._petCells);
        for (var ei = 0; ei < existingKeys.length; ei += 1) {
            var ek = existingKeys[ei];
            var cell = this._petCells[ek];
            var curVeh = curVehMap[ek];
            var curVru = curVruMap[ek];

            if (cell.vehActiveId !== null && (!curVeh || curVeh.id !== cell.vehActiveId)) {
                cell.vehLastExitId = cell.vehActiveId;
                cell.vehLastExitTime = prevTimeS;
                cell.vehActiveId = null;
            }
            if (cell.vruActiveId !== null && (!curVru || curVru.id !== cell.vruActiveId)) {
                cell.vruLastExitId = cell.vruActiveId;
                cell.vruLastExitTime = prevTimeS;
                cell.vruActiveId = null;
            }
        }

        var touchedKeys = Object.keys(curVehMap);
        var vruKeys = Object.keys(curVruMap);
        for (var vi2 = 0; vi2 < vruKeys.length; vi2 += 1) {
            if (!curVehMap[vruKeys[vi2]]) touchedKeys.push(vruKeys[vi2]);
        }

        for (var ti = 0; ti < touchedKeys.length; ti += 1) {
            var tk = touchedKeys[ti];
            var cv = curVehMap[tk] || null;
            var cu = curVruMap[tk] || null;
            var rec = this._petCells[tk];
            if (!rec) {
                var coord = cv || cu;
                rec = {
                    cx: coord.cx,
                    cy: coord.cy,
                    vehActiveId: null,
                    vruActiveId: null,
                    vehLastExitId: null,
                    vehLastExitTime: -999,
                    vruLastExitId: null,
                    vruLastExitTime: -999,
                    vehTime: -999,
                    vruTime: -999,
                    petS: null,
                    petVehId: null,
                    petVruId: null,
                    petTime: -999
                };
                this._petCells[tk] = rec;
            }

            if (cv && cu) {
                // 车辆与 VRU 同帧同时占用该格属于并行重叠，不记作先后通行的正值 PET
                rec.vehActiveId = cv.id;
                rec.vruActiveId = cu.id;
                rec.vehTime = timeS;
                rec.vruTime = timeS;
            } else if (cv) {
                rec.vehActiveId = cv.id;
                rec.vehTime = timeS;
                if (rec.vruActiveId === null && rec.vruLastExitId !== null && timeS > rec.vruLastExitTime) {
                    var dtV = timeS - rec.vruLastExitTime;
                    if (dtV > 0 && dtV <= 2.2 && (rec.petS === null || dtV < rec.petS)) {
                        rec.petS = dtV;
                        rec.petVehId = cv.id;
                        rec.petVruId = rec.vruLastExitId;
                        rec.petTime = timeS;
                    }
                }
            } else if (cu) {
                rec.vruActiveId = cu.id;
                rec.vruTime = timeS;
                if (rec.vehActiveId === null && rec.vehLastExitId !== null && timeS > rec.vehLastExitTime) {
                    var dtU = timeS - rec.vehLastExitTime;
                    if (dtU > 0 && dtU <= 2.4 && (rec.petS === null || dtU < rec.petS)) {
                        rec.petS = dtU;
                        rec.petVehId = rec.vehLastExitId;
                        rec.petVruId = cu.id;
                        rec.petTime = timeS;
                    }
                }
            }
        }

        // Prune cells older than 3.6s to keep map clean and bounded
        var keys = Object.keys(this._petCells);
        for (var k = 0; k < keys.length; k += 1) {
            var cRec = this._petCells[keys[k]];
            if (timeS - Math.max(cRec.vehTime, cRec.vruTime, cRec.vehLastExitTime, cRec.vruLastExitTime) > 3.6) {
                delete this._petCells[keys[k]];
            }
        }
        this._lastPetTimeS = timeS;
    };

    CampusRoadTwinCanvas.prototype._evaluateSimulation = function (simData, timeS, cycleIdx) {
        var conflicts = [];
        var driverAlerts = [];
        var pedestrianAlerts = [];
        var vehMap = Object.create(null);
        var maxRisk = 0;
        var maxSpeed = 0;
        var minTtc = null;
        var cyclistVruEnabled = !(this._config && this._config.cyclistVru === false);
        var useCvmCircle = Boolean(this._config && this._config.conflictModel === "cvm_circle");
        var speedModel = (this._config && this._config.speedModel) || "ctrv";
        var activeVrus = cyclistVruEnabled
            ? simData.vrus
            : simData.vrus.filter(function (item) { return !isCyclistVruKind(item); });

        this._recordVehTrails(simData.vehicles, timeS);
        var pairPet = Object.create(null);
        if (useCvmCircle) {
            this._petCells = Object.create(null);
            this._lastPetTimeS = null;
        } else {
            this._recordPetGrid(simData.vehicles, activeVrus, timeS);
            var cellKeys = Object.keys(this._petCells);
            for (var c = 0; c < cellKeys.length; c += 1) {
                var cell = this._petCells[cellKeys[c]];
                if (cell.petS !== null && cell.petVehId !== null && cell.petVruId !== null && (timeS - cell.petTime <= 1.8)) {
                    var pk = cell.petVehId + ":" + cell.petVruId;
                    if (pairPet[pk] === undefined || cell.petS < pairPet[pk]) {
                        pairPet[pk] = cell.petS;
                    }
                }
            }
        }

        for (var vi = 0; vi < simData.vehicles.length; vi += 1) {
            var veh = simData.vehicles[vi];
            if (veh.speedKmh > maxSpeed) maxSpeed = veh.speedKmh;
            if (veh.inParkingZone && veh.speedKmh <= 5.0) {
                veh.riskLevel = 0;
                continue;
            }

            var bestConflict = null;
            for (var pi = 0; pi < activeVrus.length; pi += 1) {
                var vru = activeVrus[pi];
                var res = useCvmCircle
                    ? computeCvmCircleTtc(veh, vru, 3.5)
                    : computeCpaAndTtc(veh, vru, 3.5, speedModel);
                var pk2 = veh.trackId + ":" + vru.trackId;
                var petS = !useCvmCircle && pairPet[pk2] !== undefined ? pairPet[pk2] : null;
                if (res.currentDist > 16.0 && petS === null) continue;

                var level = 0;
                if ((res.ttcSeconds !== null && res.ttcSeconds <= 1.8) || (petS !== null && petS <= 1.0)) {
                    level = 3;
                } else if ((res.ttcSeconds !== null && res.ttcSeconds <= 3.5) || (petS !== null && petS <= 2.0)) {
                    level = 2;
                }

                if (level > 0) {
                    var cand = {
                        vehTrackId: veh.trackId,
                        pedTrackId: vru.trackId,
                        vehicleKind: veh.kind || "car",
                        targetKind: vru.kind || "pedestrian",
                        riskLevel: level,
                        vehSpeedKmh: veh.speedKmh,
                        currentDistanceM: res.currentDist,
                        minDistanceM: res.minDistanceM,
                        ttcSeconds: res.ttcSeconds,
                        petSeconds: petS,
                        conflictGroundXY: res.conflictGroundXY
                    };
                    if (!bestConflict ||
                        cand.riskLevel > bestConflict.riskLevel ||
                        (cand.riskLevel === bestConflict.riskLevel &&
                         (cand.ttcSeconds !== null ? cand.ttcSeconds : 99) < (bestConflict.ttcSeconds !== null ? bestConflict.ttcSeconds : 99))) {
                        bestConflict = cand;
                    }
                }
            }

            if (bestConflict) {
                veh.riskLevel = bestConflict.riskLevel;
                vehMap[veh.trackId] = bestConflict;
                conflicts.push(bestConflict);
                if (bestConflict.riskLevel > maxRisk) maxRisk = bestConflict.riskLevel;
                if (bestConflict.ttcSeconds !== null && (minTtc === null || bestConflict.ttcSeconds < minTtc)) {
                    minTtc = bestConflict.ttcSeconds;
                }

                var ttcTxt = bestConflict.ttcSeconds !== null ? ("TTC=" + bestConflict.ttcSeconds.toFixed(2) + "s") : "PET=" + bestConflict.petSeconds.toFixed(2) + "s";
                var spdTxt = bestConflict.vehSpeedKmh.toFixed(1) + " km/h";
                var targetLabel = (bestConflict.targetKind === "bicycle" || bestConflict.targetKind === "motorcycle") ? "骑行者" : "行人";
                var urgency = bestConflict.riskLevel === 3 ? "critical" : "caution";

                driverAlerts.push({
                    target: "driver",
                    role: "driver",
                    urgency: urgency,
                    risk_level: bestConflict.riskLevel,
                    risk_name: RISK_NAMES[bestConflict.riskLevel],
                    message: (bestConflict.riskLevel === 3 ? "紧急碰撞预警" : "注意预警") + "：车辆 #" + bestConflict.vehTrackId + " (" + spdTxt + ") 与" + targetLabel + " #" + bestConflict.pedTrackId + " 轨迹汇聚（" + ttcTxt + "）。",
                    advice: bestConflict.riskLevel === 3 ? "请立即紧急制动停车避让。" : "请提前备刹减速，注意观察前方弱势交通参与者动态。",
                    veh_track_id: bestConflict.vehTrackId,
                    ped_track_id: bestConflict.pedTrackId,
                    vehicle_kind: bestConflict.vehicleKind,
                    target_kind: bestConflict.targetKind,
                    veh_speed_kmh: Number(bestConflict.vehSpeedKmh.toFixed(1)),
                    current_distance_m: Number(bestConflict.currentDistanceM.toFixed(2)),
                    min_distance_m: Number(bestConflict.minDistanceM.toFixed(2)),
                    ttc_seconds: bestConflict.ttcSeconds !== null ? Number(bestConflict.ttcSeconds.toFixed(2)) : null,
                    pet_seconds: bestConflict.petSeconds !== null ? Number(bestConflict.petSeconds.toFixed(2)) : null
                });

                pedestrianAlerts.push({
                    target: "pedestrian",
                    role: "pedestrian",
                    urgency: urgency,
                    risk_level: bestConflict.riskLevel,
                    risk_name: RISK_NAMES[bestConflict.riskLevel],
                    message: (bestConflict.riskLevel === 3 ? "紧急避险预警" : "路旁提示") + "：" + targetLabel + " #" + bestConflict.pedTrackId + " 请注意车辆 #" + bestConflict.vehTrackId + " (" + spdTxt + ") 正在接近（" + ttcTxt + "）。",
                    advice: bestConflict.riskLevel === 3 ? "请立即止步并向路侧安全区域退让避险。" : "请停步观望或沿安全区域通行，避免盲目横穿。",
                    veh_track_id: bestConflict.vehTrackId,
                    ped_track_id: bestConflict.pedTrackId,
                    vehicle_kind: bestConflict.vehicleKind,
                    target_kind: bestConflict.targetKind,
                    veh_speed_kmh: Number(bestConflict.vehSpeedKmh.toFixed(1)),
                    current_distance_m: Number(bestConflict.currentDistanceM.toFixed(2)),
                    min_distance_m: Number(bestConflict.minDistanceM.toFixed(2)),
                    ttc_seconds: bestConflict.ttcSeconds !== null ? Number(bestConflict.ttcSeconds.toFixed(2)) : null,
                    pet_seconds: bestConflict.petSeconds !== null ? Number(bestConflict.petSeconds.toFixed(2)) : null
                });
            } else if (veh.speedKmh > this._speedLimitKmh) {
                veh.riskLevel = 1;
                if (1 > maxRisk) maxRisk = 1;
                driverAlerts.push({
                    target: "driver",
                    role: "driver",
                    urgency: "speeding",
                    risk_level: 1,
                    risk_name: RISK_NAMES[1],
                    message: "超速提示：车辆 #" + veh.trackId + " (" + (veh.kind || "motor_vehicle") + ") 当前车速 " + veh.speedKmh.toFixed(1) + " km/h，已超出校园限速。",
                    advice: "请立即减速慢行，将车速控制在校内限速以内。",
                    veh_track_id: veh.trackId,
                    ped_track_id: null,
                    vehicle_kind: veh.kind || "motor_vehicle",
                    target_kind: "pedestrian",
                    veh_speed_kmh: Number(veh.speedKmh.toFixed(1)),
                    current_distance_m: 0,
                    min_distance_m: 0,
                    ttc_seconds: null,
                    pet_seconds: null
                });
            } else {
                veh.riskLevel = 0;
            }
        }

        // Tag VRUs involved in conflicts for clean ring highlight
        for (var j = 0; j < simData.vrus.length; j += 1) {
            var item = simData.vrus[j];
            item.riskLevel = 0;
            for (var cIdx = 0; cIdx < conflicts.length; cIdx += 1) {
                if (conflicts[cIdx].pedTrackId === item.trackId && conflicts[cIdx].riskLevel > item.riskLevel) {
                    item.riskLevel = conflicts[cIdx].riskLevel;
                }
            }
        }

        var cycleTimestamp = Number(timeS.toFixed(2));
        this._latestSimSnapshot = {
            timestampS: cycleTimestamp,
            cycleDurationS: SCENARIO_CYCLE_S,
            cycleIndex: cycleIdx,
            fps: 25,
            frameCount: this._simFrameCount,
            maxSpeedKmh: Number(maxSpeed.toFixed(1)),
            minTtcS: minTtc !== null ? Number(minTtc.toFixed(2)) : null,
            activeConflicts: driverAlerts.length,
            riskLevel: maxRisk,
            riskName: RISK_NAMES[maxRisk],
            driverAlerts: driverAlerts,
            pedestrianAlerts: pedestrianAlerts,
            vehicleCount: simData.vehicles.length,
            vruCount: simData.vrus.length,
            seriesPoint: {
                t: cycleTimestamp,
                speedKmh: Number(maxSpeed.toFixed(1)),
                ttcS: minTtc !== null ? Number(minTtc.toFixed(2)) : null,
                riskLevel: maxRisk
            }
        };

        this._currentRenderState = {
            bounds: simData.bounds,
            parkingZones: simData.parkingZones,
            vehicles: simData.vehicles,
            vrus: simData.vrus,
            conflicts: conflicts,
            label: this._scenarioTitle(this._scenarioId),
            simTimeS: timeS,
            maxRisk: maxRisk,
            speedModel: speedModel,
            conflictModel: useCvmCircle ? "cvm_circle" : "obb_ssm",
            cycleFade: simData.cycleFade !== undefined ? simData.cycleFade : 1.0
        };
    };

    CampusRoadTwinCanvas.prototype._scenarioTitle = function (id) {
        if (id === "cyclist_vru_cutin") return "SIM · 骑行者盲区切入";
        if (id === "ctrv_curve_yield") return "SIM · 弯道 CTRV 轨迹会车";
        return "SIM · 斑马线横穿急刹";
    };

    CampusRoadTwinCanvas.prototype._stepSimulation = function (dt, forceEmit, skipRender) {
        if (this._mode !== "simulation") return;
        var prevCycleIdx = Math.floor(this._simTimeS / SCENARIO_CYCLE_S);
        this._simTimeS += dt;
        var cycleIdx = Math.floor(this._simTimeS / SCENARIO_CYCLE_S);
        if (cycleIdx !== prevCycleIdx) {
            // Reset PET grid and swept trails cleanly on scenario cycle restart
            this._petCells = Object.create(null);
            this._lastPetTimeS = null;
            this._vehTrails = Object.create(null);
        }
        var cycleTime = ((this._simTimeS % SCENARIO_CYCLE_S) + SCENARIO_CYCLE_S) % SCENARIO_CYCLE_S;
        this._simFrameCount = Math.min(Math.round(SCENARIO_CYCLE_S * 25), Math.floor(cycleTime * 25 + 1e-9) + 1);
        var simData = buildScenarioState(this._scenarioId, cycleTime, this._config);
        this._evaluateSimulation(simData, cycleTime, cycleIdx);
        if (!skipRender) {
            this._renderCurrent();
        }
        this._emitSimTelemetry(Boolean(forceEmit));
    };

    CampusRoadTwinCanvas.prototype._emitSimTelemetry = function (force) {
        if (!this._onTelemetry || !this._latestSimSnapshot || this._mode !== "simulation") {
            return;
        }
        var snap = this._latestSimSnapshot;
        var stateChanged = snap.riskLevel !== this._lastEmittedRisk ||
            snap.activeConflicts !== this._lastEmittedAlerts;
        var now = typeof performance !== "undefined" ? performance.now() : Date.now();
        if (force || stateChanged || (now - this._lastTelemetryTs >= 100)) {
            this._lastTelemetryTs = now;
            this._lastEmittedRisk = snap.riskLevel;
            this._lastEmittedAlerts = snap.activeConflicts;
            this._onTelemetry(snap);
        }
    };

    CampusRoadTwinCanvas.prototype._stopVisibleLoop = function () {
        if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
            global.cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
        this._lastDrawTs = null;
    };

    CampusRoadTwinCanvas.prototype._stopBgLoop = function () {
        if (this._bgTimerId !== null && typeof global.clearInterval === "function") {
            global.clearInterval(this._bgTimerId);
            this._bgTimerId = null;
        }
        this._lastBgTs = null;
    };

    CampusRoadTwinCanvas.prototype._stopLoop = function () {
        this._stopVisibleLoop();
        this._stopBgLoop();
    };

    CampusRoadTwinCanvas.prototype._syncLoop = function () {
        var simActive = !this._destroyed &&
            this._mode === "simulation" &&
            !this._userPaused &&
            !this._motionTracker.isReduced();

        if (!simActive) {
            this._stopLoop();
            return;
        }

        if (this._offscreenPaused) {
            // Keep physical simulation + 8Hz telemetry active while canvas is offscreen (e.g. Alerts tab)
            this._stopVisibleLoop();
            if (this._bgTimerId === null && typeof global.setInterval === "function") {
                var bgSelf = this;
                this._lastBgTs = typeof performance !== "undefined" ? performance.now() : Date.now();
                this._bgTimerId = global.setInterval(function () {
                    if (bgSelf._destroyed ||
                        bgSelf._mode !== "simulation" ||
                        bgSelf._userPaused ||
                        !bgSelf._offscreenPaused ||
                        bgSelf._motionTracker.isReduced()) {
                        bgSelf._stopBgLoop();
                        return;
                    }
                    var now = typeof performance !== "undefined" ? performance.now() : Date.now();
                    var dt = Math.min(0.25, Math.max(0.02, (now - (bgSelf._lastBgTs || now)) / 1000));
                    bgSelf._lastBgTs = now;
                    bgSelf._stepSimulation(dt, true, true);
                }, 125);
            }
            return;
        }

        this._stopBgLoop();
        if (this._rafId !== null || typeof global.requestAnimationFrame !== "function") {
            return;
        }

        var self = this;
        function frame(now) {
            if (self._destroyed ||
                self._mode !== "simulation" ||
                self._userPaused ||
                self._offscreenPaused ||
                self._motionTracker.isReduced()) {
                self._rafId = null;
                return;
            }
            if (self._lastDrawTs === null) {
                self._lastDrawTs = now;
            }
            var dt = Math.min(0.06, Math.max(0, (now - self._lastDrawTs) / 1000));
            self._lastDrawTs = now;
            self._stepSimulation(dt, false, false);
            self._rafId = global.requestAnimationFrame(frame);
        }

        this._rafId = global.requestAnimationFrame(frame);
    };

    CampusRoadTwinCanvas.prototype._renderCurrent = function () {
        if (!this._ctx || this._cssWidth < 2 || this._cssHeight < 2) return;
        var ctx = this._ctx;
        var pal = this._palette;
        var st = this._currentRenderState;
        var w = this._cssWidth;
        var h = this._cssHeight;

        ctx.save();
        ctx.scale(this._dpr, this._dpr);
        ctx.clearRect(0, 0, w, h);

        // Background shoulder
        ctx.fillStyle = pal.roadShoulder;
        ctx.fillRect(0, 0, w, h);

        var padL = 44;
        var padR = 20;
        var padT = 32;
        var padB = 26;
        var availW = Math.max(40, w - padL - padR);
        var availH = Math.max(40, h - padT - padB);

        var b = st.bounds;
        var spanX = Math.max(2, b.xMax - b.xMin);
        var spanY = Math.max(2, b.yMax - b.yMin);
        var scale = Math.min(availW / spanX, availH / spanY);
        var drawW = spanX * scale;
        var drawH = spanY * scale;
        var originX = padL + (availW - drawW) * 0.5;
        var originY = padT + (availH - drawH) * 0.5 + drawH; // yMin at bottom, yMax at top

        function toPx(wx, wy) {
            return {
                x: originX + (wx - b.xMin) * scale,
                y: originY - (wy - b.yMin) * scale
            };
        }

        // Roadbed polygon & calibrated 1:1 sidewalk shoulders
        var roadTopLeft = toPx(b.xMin, b.yMax);
        var roadBotRight = toPx(b.xMax, b.yMin);

        var leftSidewalkW = Math.max(0, roadTopLeft.x - padL);
        var rightSidewalkW = Math.max(0, (w - padR) - roadBotRight.x);
        var sidewalkFill = withAlpha(pal.bgCard, pal.isLight ? 0.56 : 0.40);
        var tileStroke = withAlpha(pal.hairline, pal.isLight ? 0.70 : 0.65);

        if (leftSidewalkW > 4) {
            ctx.fillStyle = sidewalkFill;
            ctx.fillRect(padL, roadTopLeft.y, leftSidewalkW, drawH);
        }
        if (rightSidewalkW > 4) {
            ctx.fillStyle = sidewalkFill;
            ctx.fillRect(roadBotRight.x, roadTopLeft.y, rightSidewalkW, drawH);
        }

        // Subtle architectural sidewalk paving joints (clipped strictly to left/right sidewalk bands)
        if (leftSidewalkW > 12 || rightSidewalkW > 12) {
            ctx.save();
            ctx.strokeStyle = tileStroke;
            ctx.lineWidth = 1;
            var slabStepM = 1.8;
            for (var sy = Math.ceil(b.yMin / slabStepM) * slabStepM; sy < b.yMax; sy += slabStepM) {
                var syPx = toPx(b.xMin, sy).y;
                if (leftSidewalkW > 12) {
                    ctx.beginPath();
                    ctx.moveTo(padL, syPx);
                    ctx.lineTo(roadTopLeft.x, syPx);
                    ctx.stroke();
                }
                if (rightSidewalkW > 12) {
                    ctx.beginPath();
                    ctx.moveTo(roadBotRight.x, syPx);
                    ctx.lineTo(w - padR, syPx);
                    ctx.stroke();
                }
            }
            var swColM = 2.0 * scale;
            if (leftSidewalkW > swColM) {
                for (var lx = roadTopLeft.x - swColM; lx > padL + 6; lx -= swColM) {
                    ctx.beginPath();
                    ctx.moveTo(lx, roadTopLeft.y);
                    ctx.lineTo(lx, roadBotRight.y);
                    ctx.stroke();
                }
            }
            if (rightSidewalkW > swColM) {
                for (var rx = roadBotRight.x + swColM; rx < w - padR - 6; rx += swColM) {
                    ctx.beginPath();
                    ctx.moveTo(rx, roadTopLeft.y);
                    ctx.lineTo(rx, roadBotRight.y);
                    ctx.stroke();
                }
            }
            ctx.restore();
        }

        ctx.fillStyle = pal.roadSurface;
        ctx.fillRect(roadTopLeft.x, roadTopLeft.y, roadBotRight.x - roadTopLeft.x, roadBotRight.y - roadTopLeft.y);

        // 1m subtle ruler grid + distance labels
        ctx.lineWidth = 1;
        ctx.font = "500 10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

        var firstX = Math.ceil(b.xMin);
        var xLabelStep = scale < 36 ? Math.ceil(36 / Math.max(1, scale)) : 1;
        for (var gx = firstX; gx <= Math.floor(b.xMax); gx += 1) {
            var px1 = toPx(gx, b.yMin);
            var px2 = toPx(gx, b.yMax);
            ctx.strokeStyle = pal.gridLine;
            ctx.beginPath();
            ctx.moveTo(px1.x, px1.y);
            ctx.lineTo(px2.x, px2.y);
            ctx.stroke();

            if (gx % xLabelStep === 0) {
                ctx.fillStyle = pal.textMuted;
                ctx.textAlign = "center";
                ctx.textBaseline = "top";
                ctx.fillText(gx + "m", px1.x, roadBotRight.y + 6);
            }
        }

        var firstY = Math.ceil(b.yMin);
        var yStep = spanY > 12 ? 2 : 1;
        for (var gy = firstY; gy <= Math.floor(b.yMax); gy += 1) {
            var py1 = toPx(b.xMin, gy);
            var py2 = toPx(b.xMax, gy);
            ctx.strokeStyle = pal.gridLine;
            ctx.beginPath();
            ctx.moveTo(py1.x, py1.y);
            ctx.lineTo(py2.x, py2.y);
            ctx.stroke();

            if (gy % yStep === 0 || gy === Math.floor(b.yMax)) {
                ctx.fillStyle = pal.textMuted;
                ctx.textAlign = "right";
                ctx.textBaseline = "middle";
                ctx.fillText(gy + "m", padL - 6, py1.y);
            }
        }

        // Crosswalk zebra stripes strictly inside curbs (y = 8.80 .. 10.00m)
        var cwY0 = 8.80;
        var cwY1 = 10.00;
        if (cwY1 >= b.yMin + 0.4 && cwY0 <= b.yMax - 0.4) {
            var cwLeft = b.xMin + 0.35;
            var cwRight = b.xMax - 0.35;
            var cwClampedY0 = Math.max(b.yMin, cwY0);
            var cwClampedY1 = Math.min(b.yMax, cwY1);
            ctx.fillStyle = withAlpha(pal.textPrimary, pal.isLight ? 0.14 : 0.11);
            for (var sx = cwLeft; sx < cwRight - 0.25; sx += 0.65) {
                var spA = toPx(sx, cwClampedY1);
                var spB = toPx(Math.min(sx + 0.34, cwRight), cwClampedY0);
                ctx.fillRect(spA.x, spA.y, spB.x - spA.x, spB.y - spA.y);
            }
        }

        // Road boundary curbs and center dashed lane divider
        var curbInset = 0.25;
        var curbL0 = toPx(b.xMin + curbInset, b.yMin);
        var curbL1 = toPx(b.xMin + curbInset, b.yMax);
        var curbR0 = toPx(b.xMax - curbInset, b.yMin);
        var curbR1 = toPx(b.xMax - curbInset, b.yMax);
        ctx.strokeStyle = pal.roadCurb;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(curbL0.x, curbL0.y);
        ctx.lineTo(curbL1.x, curbL1.y);
        ctx.moveTo(curbR0.x, curbR0.y);
        ctx.lineTo(curbR1.x, curbR1.y);
        ctx.stroke();

        var centerX = 0;
        if (centerX <= b.xMin + 0.8 || centerX >= b.xMax - 0.8) {
            centerX = 0.5 * (b.xMin + b.xMax);
        }
        var div0 = toPx(centerX, b.yMin);
        var div1 = toPx(centerX, b.yMax);
        ctx.save();
        ctx.setLineDash([8, 8]);
        ctx.strokeStyle = pal.roadMark;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(div0.x, div0.y);
        ctx.lineTo(div1.x, div1.y);
        ctx.stroke();
        ctx.restore();

        // Collision-free pill reservation tracking
        var placedBoxes = [];
        var obstacleBoxes = [];
        var headerLabel = st.label || (this._mode === "replay" ? "REPLAY" : "SIMULATION");
        ctx.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        var headerW = ctx.measureText(headerLabel).width + 12;
        placedBoxes.push({ x: 10, y: 8, w: headerW, h: 18 });

        var safeArea = {
            minX: padL + 2,
            maxX: w - 8,
            minY: Math.max(32, roadTopLeft.y),
            maxY: Math.max(54, roadBotRight.y - 6)
        };

        // Precompute OBB footprint screen boxes so labels avoid sitting on top of cars/VRUs
        for (var oi = 0; oi < st.vehicles.length; oi += 1) {
            var ov = st.vehicles[oi];
            var ovPx = toPx(ov.x, ov.y);
            var oLen = Math.max(14, (ov.lengthM || 4.5) * scale);
            var oWid = Math.max(8, (ov.widthM || 1.9) * scale);
            var ovHead = resolveHeadingRad(ov.headingRad, Math.PI * 0.5);
            var cosA = Math.abs(Math.cos(ovHead));
            var sinA = Math.abs(Math.sin(ovHead));
            var bW = oLen * cosA + oWid * sinA;
            var bH = oLen * sinA + oWid * cosA;
            obstacleBoxes.push({ x: ovPx.x - bW * 0.5, y: ovPx.y - bH * 0.5, w: bW, h: bH });
        }

        // Roadside parking exclusion zone polygon
        var zones = st.parkingZones || [];
        for (var zi = 0; zi < zones.length; zi += 1) {
            var poly = zones[zi];
            if (!Array.isArray(poly) || poly.length < 3) continue;
            var pMinX = Infinity, pMaxX = -Infinity, pMinY = Infinity, pMaxY = -Infinity;
            ctx.save();
            ctx.beginPath();
            for (var pi = 0; pi < poly.length; pi += 1) {
                var pPt = toPx(poly[pi][0], poly[pi][1]);
                if (pPt.x < pMinX) pMinX = pPt.x;
                if (pPt.x > pMaxX) pMaxX = pPt.x;
                if (pPt.y < pMinY) pMinY = pPt.y;
                if (pPt.y > pMaxY) pMaxY = pPt.y;
                if (pi === 0) ctx.moveTo(pPt.x, pPt.y);
                else ctx.lineTo(pPt.x, pPt.y);
            }
            ctx.closePath();
            ctx.fillStyle = withAlpha(pal.parkingZone, 0.08);
            ctx.fill();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = withAlpha(pal.parkingZone, 0.72);
            ctx.lineWidth = 1.2;
            ctx.stroke();
            ctx.restore();

            ctx.font = "500 10px system-ui, -apple-system, sans-serif";
            var pzText = "禁停标定区";
            var pzW = ctx.measureText(pzText).width + 12;
            var pzH = 18;
            var pzCand = [
                { x: pMaxX - pzW - 2, y: pMinY - pzH - 4 },
                { x: pMinX + 2, y: pMinY - pzH - 4 },
                { x: pMaxX + 6, y: pMinY + 2 },
                { x: pMinX + 3, y: pMaxY - pzH - 2 }
            ];
            var pzPicked = chooseBestBox(pzCand, pzW, pzH, safeArea, placedBoxes, obstacleBoxes);
            drawRoundedRectPath(ctx, pzPicked.x, pzPicked.y, pzPicked.w, pzPicked.h, 4);
            ctx.fillStyle = withAlpha(pal.bgCard, 0.92);
            ctx.fill();
            ctx.lineWidth = 1;
            ctx.strokeStyle = withAlpha(pal.parkingZone, 0.72);
            ctx.stroke();
            ctx.fillStyle = pal.parkingZone;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(pzText, pzPicked.x + 6, pzPicked.y + pzPicked.h * 0.5 + 0.5);
            placedBoxes.push(pzPicked);
        }

        // Clip dynamic road overlays (PET conflict cells, swept ribbons, CTRV arcs, TTC rings) strictly inside road network draw area
        var netClipX = padL;
        var netClipY = Math.max(30, roadTopLeft.y);
        var netClipW = Math.max(40, (w - padR) - padL);
        var netClipH = Math.max(20, roadBotRight.y - netClipY);

        ctx.save();
        ctx.beginPath();
        ctx.rect(netClipX, netClipY, netClipW, netClipH);
        ctx.clip();

        // Subtle translucent swept centerline ribbon behind moving vehicles
        var trailList = st.vehicles || [];
        for (var trIdx = 0; trIdx < trailList.length; trIdx += 1) {
            var trVeh = trailList[trIdx];
            if (trVeh.inParkingZone) continue;
            var trailPts = this._vehTrails[String(trVeh.trackId)];
            if (!trailPts || trailPts.length < 2) continue;
            var trColor = trVeh.riskLevel >= 1 ? riskColor(pal, trVeh.riskLevel) : pal.safe;
            var ribbonW = Math.max(6, (trVeh.widthM || 1.9) * scale * 0.42);
            ctx.save();
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.beginPath();
            for (var tp = 0; tp < trailPts.length; tp += 1) {
                var trPx = toPx(trailPts[tp].x, trailPts[tp].y);
                if (tp === 0) ctx.moveTo(trPx.x, trPx.y);
                else ctx.lineTo(trPx.x, trPx.y);
            }
            ctx.strokeStyle = withAlpha(trColor, 0.11);
            ctx.lineWidth = ribbonW;
            ctx.stroke();
            ctx.strokeStyle = withAlpha(trColor, 0.32);
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.restore();
        }

        // Discrete 0.5m PET conflict intersection cells ONLY (no muddy raw occupancy grid squares)
        var refTime = st.simTimeS;
        var cellSize = 0.5;
        var cellKeys = Object.keys(this._petCells);
        for (var ci = 0; ci < cellKeys.length; ci += 1) {
            var cell = this._petCells[cellKeys[ci]];
            if (cell.petS === null) continue;
            var latestTouch = Math.max(cell.vehTime, cell.vruTime);
            var age = refTime - latestTouch;
            if (age < 0 || age > 2.4) continue;
            var wx0 = cell.cx * cellSize;
            var wy0 = cell.cy * cellSize;
            if (wx0 + cellSize < b.xMin || wx0 > b.xMax || wy0 + cellSize < b.yMin || wy0 > b.yMax) continue;

            var cTopLeft = toPx(wx0, wy0 + cellSize);
            var cBotRight = toPx(wx0 + cellSize, wy0);
            var cw = cBotRight.x - cTopLeft.x;
            var ch = cBotRight.y - cTopLeft.y;
            var fade = clamp(1 - age / 2.4, 0.18, 1);

            var petUrgent = cell.petS <= 1.0;
            var petTint = petUrgent ? pal.critical : pal.accent;
            ctx.fillStyle = withAlpha(petTint, 0.24 * fade);
            ctx.fillRect(cTopLeft.x + 0.5, cTopLeft.y + 0.5, cw - 1, ch - 1);
            ctx.strokeStyle = withAlpha(petTint, 0.68 * fade);
            ctx.lineWidth = 1;
            ctx.strokeRect(cTopLeft.x + 0.5, cTopLeft.y + 0.5, cw - 1, ch - 1);
        }

        // Draw kinematic prediction arcs beneath vehicles (honoring active speedModel)
        for (var vi = 0; vi < st.vehicles.length; vi += 1) {
            var veh = st.vehicles[vi];
            var spdMs = Math.hypot(veh.vx || 0, veh.vy || 0);
            if (spdMs < 0.35 || veh.inParkingZone) continue;

            var arcPts = predictCtrvArc(veh, 2.5, 16, st.speedModel);
            var arcColor = veh.riskLevel >= 1 ? riskColor(pal, veh.riskLevel) : pal.safe;
            ctx.save();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = withAlpha(arcColor, 0.78);
            ctx.lineWidth = 1.6;
            ctx.beginPath();
            for (var ai = 0; ai < arcPts.length; ai += 1) {
                var aPx = toPx(arcPts[ai].x, arcPts[ai].y);
                if (ai === 0) ctx.moveTo(aPx.x, aPx.y);
                else ctx.lineTo(aPx.x, aPx.y);
            }
            ctx.stroke();
            ctx.restore();

            // Subtle 0.5s trajectory tick markers
            ctx.fillStyle = withAlpha(arcColor, 0.85);
            for (var ti = 3; ti < arcPts.length; ti += 3) {
                var tPx = toPx(arcPts[ti].x, arcPts[ti].y);
                ctx.beginPath();
                ctx.arc(tPx.x, tPx.y, 2, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Draw TTC convergence links & calm concentric arcs; defer pill drawing for priority collision avoidance
        var pendingConflictPills = [];
        for (var k = 0; k < st.conflicts.length; k += 1) {
            var cf = st.conflicts[k];
            if ((cf.riskLevel || 0) < 2) continue;
            var cfColor = riskColor(pal, cf.riskLevel);
            var vehObj = findById(st.vehicles, cf.vehTrackId);
            var vruObj = findById(st.vrus, cf.pedTrackId);

            if (vehObj && vruObj) {
                var pA = toPx(vehObj.x, vehObj.y);
                var pB = toPx(vruObj.x, vruObj.y);
                ctx.save();
                ctx.setLineDash([3, 3]);
                ctx.strokeStyle = withAlpha(cfColor, 0.75);
                ctx.lineWidth = 1.25;
                ctx.beginPath();
                ctx.moveTo(pA.x, pA.y);
                ctx.lineTo(pB.x, pB.y);
                ctx.stroke();
                ctx.restore();
            }

            var cxy = cf.conflictGroundXY;
            if ((!Array.isArray(cxy) || cxy.length < 2) && vehObj && vruObj) {
                cxy = [(vehObj.x + vruObj.x) * 0.5, (vehObj.y + vruObj.y) * 0.5];
            }
            if (Array.isArray(cxy) && cxy.length >= 2 && isFiniteNum(cxy[0]) && isFiniteNum(cxy[1])) {
                var cPx = toPx(cxy[0], cxy[1]);
                ctx.strokeStyle = withAlpha(cfColor, 0.65);
                ctx.lineWidth = 1.3;
                ctx.beginPath();
                ctx.arc(cPx.x, cPx.y, 10, 0, Math.PI * 2);
                ctx.stroke();

                ctx.strokeStyle = withAlpha(cfColor, 0.28);
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(cPx.x, cPx.y, 17, 0, Math.PI * 2);
                ctx.stroke();

                var pillParts = [];
                if (isFiniteNum(cf.ttcSeconds)) pillParts.push("TTC " + cf.ttcSeconds.toFixed(1) + "s");
                if (isFiniteNum(cf.petSeconds)) pillParts.push("PET " + cf.petSeconds.toFixed(1) + "s");
                else if (isFiniteNum(cf.minDistanceM)) pillParts.push("Dmin " + cf.minDistanceM.toFixed(1) + "m");

                if (pillParts.length > 0) {
                    pendingConflictPills.push({
                        cx: cPx.x,
                        cy: cPx.y,
                        rx: 18,
                        ry: 18,
                        text: pillParts.join(" / "),
                        border: cfColor
                    });
                }
            }
        }

        // Draw rounded OBB vehicles (clipped inside calibrated road viewport; record pill tasks for collision-free layout)
        var pendingVehiclePills = [];
        for (var vi2 = 0; vi2 < st.vehicles.length; vi2 += 1) {
            var v = st.vehicles[vi2];
            var vPx = toPx(v.x, v.y);
            var vColor = v.inParkingZone
                ? pal.textMuted
                : (v.riskLevel >= 1 ? riskColor(pal, v.riskLevel) : pal.safe);
            var lenPx = Math.max(14, (v.lengthM || 4.5) * scale);
            var widPx = Math.max(8, (v.widthM || 1.9) * scale);
            var cornerPx = Math.min(lenPx, widPx) * 0.22;
            var vehHeading = resolveHeadingRad(v.headingRad, Math.PI * 0.5);

            ctx.save();
            if (!v.inParkingZone && isFiniteNum(st.cycleFade) && st.cycleFade < 1.0) {
                ctx.globalAlpha = clamp(st.cycleFade, 0.05, 1.0);
            }
            ctx.translate(vPx.x, vPx.y);
            ctx.rotate(-vehHeading);

            drawRoundedRectPath(ctx, -lenPx * 0.5, -widPx * 0.5, lenPx, widPx, cornerPx);
            ctx.fillStyle = withAlpha(vColor, v.inParkingZone ? 0.14 : 0.20);
            ctx.fill();
            ctx.lineWidth = v.riskLevel >= 2 ? 2.0 : 1.5;
            ctx.strokeStyle = vColor;
            ctx.stroke();

            // Crisp front windshield trapezoid
            ctx.fillStyle = withAlpha(pal.textPrimary, pal.isLight ? 0.24 : 0.20);
            ctx.beginPath();
            ctx.moveTo(lenPx * 0.14, -widPx * 0.34);
            ctx.lineTo(lenPx * 0.31, -widPx * 0.27);
            ctx.lineTo(lenPx * 0.31, widPx * 0.27);
            ctx.lineTo(lenPx * 0.14, widPx * 0.34);
            ctx.closePath();
            ctx.fill();

            // Semantic brake-light bar at rear bumper when decelerating (accelMs2 < -1.0)
            if (!v.inParkingZone && isFiniteNum(v.accelMs2) && v.accelMs2 < -1.0) {
                var brakeX = -lenPx * 0.5 + 1.2;
                ctx.fillStyle = pal.critical;
                ctx.fillRect(brakeX, -widPx * 0.34, 2.4, widPx * 0.68);
            }

            // Directional heading triangle / headlight notch at +halfLen nose
            var noseX = lenPx * 0.5;
            ctx.fillStyle = vColor;
            ctx.beginPath();
            ctx.moveTo(noseX - 2, 0);
            ctx.lineTo(noseX - 8, -widPx * 0.24);
            ctx.lineTo(noseX - 8, widPx * 0.24);
            ctx.closePath();
            ctx.fill();
            ctx.restore();

            if (v.y <= b.yMax + 0.5) {
                var spdDisplay = isFiniteNum(v.speedKmh) ? v.speedKmh.toFixed(1) + "km/h" : "0.0km/h";
                var badgeText = "V-" + padId(v.trackId) + " " + spdDisplay;
                var cosV = Math.abs(Math.cos(vehHeading));
                var sinV = Math.abs(Math.sin(vehHeading));
                pendingVehiclePills.push({
                    cx: clamp(vPx.x, safeArea.minX + 12, safeArea.maxX - 12),
                    cy: clamp(vPx.y, safeArea.minY + 12, safeArea.maxY - 12),
                    rx: (lenPx * cosV + widPx * sinV) * 0.5,
                    ry: (lenPx * sinV + widPx * cosV) * 0.5,
                    text: badgeText,
                    border: withAlpha(vColor, 0.55)
                });
            }
        }

        // Draw pedestrian / cyclist VRUs (record pill tasks for collision-free layout)
        var pendingVruPills = [];
        for (var pi2 = 0; pi2 < st.vrus.length; pi2 += 1) {
            var u = st.vrus[pi2];
            var uPx = toPx(u.x, u.y);
            var isCyclist = u.kind === "bicycle" || u.category === "bicycle" || u.kind === "motorcycle";
            var uColor = u.riskLevel >= 2 ? riskColor(pal, u.riskLevel) : (isCyclist ? pal.accent : pal.textPrimary);

            // Velocity vector
            var uvx = u.vx || 0;
            var uvy = u.vy || 0;
            var uSpd = Math.hypot(uvx, uvy);
            if (uSpd > 0.15) {
                var tip = toPx(u.x + uvx * 0.9, u.y + uvy * 0.9);
                ctx.strokeStyle = withAlpha(uColor, 0.85);
                ctx.lineWidth = 1.4;
                ctx.beginPath();
                ctx.moveTo(uPx.x, uPx.y);
                ctx.lineTo(tip.x, tip.y);
                ctx.stroke();
            }

            var vruR = Math.max(6, (u.radiusM || 0.42) * scale * 0.8);
            if (isCyclist) {
                var cLen = Math.max(12, 1.8 * scale);
                var cWid = Math.max(6, 0.72 * scale);
                vruR = Math.max(cLen, cWid) * 0.5;
                ctx.save();
                ctx.translate(uPx.x, uPx.y);
                ctx.rotate(-(u.headingRad !== undefined ? u.headingRad : Math.atan2(uvy, uvx || 1)));
                drawRoundedRectPath(ctx, -cLen * 0.5, -cWid * 0.5, cLen, cWid, cWid * 0.45);
                ctx.fillStyle = withAlpha(uColor, 0.25);
                ctx.fill();
                ctx.lineWidth = 1.6;
                ctx.strokeStyle = uColor;
                ctx.stroke();
                ctx.restore();
            } else {
                var rPx = Math.max(4.5, (u.radiusM || 0.42) * scale * 0.8);
                ctx.fillStyle = withAlpha(uColor, 0.24);
                ctx.beginPath();
                ctx.arc(uPx.x, uPx.y, rPx, 0, Math.PI * 2);
                ctx.fill();
                ctx.lineWidth = 1.6;
                ctx.strokeStyle = uColor;
                ctx.stroke();
            }

            var tagPrefix = isCyclist ? "B-" : "P-";
            pendingVruPills.push({
                cx: uPx.x,
                cy: uPx.y,
                rx: vruR,
                ry: vruR,
                text: tagPrefix + padId(u.trackId),
                border: withAlpha(uColor, 0.55)
            });
        }

        // End road network clip before placing crisp unclipped pills
        ctx.restore();

        // Top-left status pill
        drawPill(ctx, 10, 8, headerLabel, pal.bgCard, pal.hairline, pal.textSecondary, w);

        // Collision-free pill pass (priority: TTC/PET -> VRUs -> Vehicles)
        for (var cp = 0; cp < pendingConflictPills.length; cp += 1) {
            var cItem = pendingConflictPills[cp];
            placePill(ctx, cItem, cItem.text, pal.bgCard, cItem.border, pal.textPrimary, safeArea, placedBoxes, obstacleBoxes);
        }
        for (var vp = 0; vp < pendingVruPills.length; vp += 1) {
            var uItem = pendingVruPills[vp];
            placePill(ctx, uItem, uItem.text, pal.bgCard, uItem.border, pal.textPrimary, safeArea, placedBoxes, obstacleBoxes);
        }
        for (var qp = 0; qp < pendingVehiclePills.length; qp += 1) {
            var vItem = pendingVehiclePills[qp];
            placePill(ctx, vItem, vItem.text, pal.bgCard, vItem.border, pal.textPrimary, safeArea, placedBoxes, obstacleBoxes);
        }

        ctx.restore();
        if (this._onRenderState) {
            this._onRenderState(this._currentRenderState);
        }
    };

    CampusRoadTwinCanvas.prototype.getCurrentRenderState = function () {
        return this._currentRenderState;
    };

    function padId(n) {
        var s = String(n);
        return s.length < 2 ? "0" + s : s;
    }

    function findById(list, id) {
        if (id === null || id === undefined) return null;
        for (var i = 0; i < list.length; i += 1) {
            if (String(list[i].trackId) === String(id)) return list[i];
        }
        return null;
    }

    function boxOverlapArea(a, b, pad) {
        var p = pad || 0;
        var ox = Math.max(0, Math.min(a.x + a.w + p, b.x + b.w + p) - Math.max(a.x, b.x));
        var oy = Math.max(0, Math.min(a.y + a.h + p, b.y + b.h + p) - Math.max(a.y, b.y));
        return ox * oy;
    }

    function chooseBestBox(candidates, boxW, boxH, safeArea, placedBoxes, obstacleBoxes) {
        var minX = safeArea.minX;
        var maxX = Math.max(minX, safeArea.maxX - boxW);
        var minY = safeArea.minY;
        var maxY = Math.max(minY, safeArea.maxY - boxH);

        var expanded = [];
        var yNudges = [0, -22, 22, -44, 44];
        for (var n = 0; n < yNudges.length; n += 1) {
            for (var c = 0; c < candidates.length; c += 1) {
                var cx = clamp(candidates[c].x, minX, maxX);
                var cy = clamp(candidates[c].y + yNudges[n], minY, maxY);
                expanded.push({ x: cx, y: cy, w: boxW, h: boxH, nudge: Math.abs(yNudges[n]) + c * 2 });
            }
        }

        var best = expanded[0];
        var bestScore = Infinity;
        for (var i = 0; i < expanded.length; i += 1) {
            var cand = expanded[i];
            var pillHit = 0;
            for (var j = 0; j < placedBoxes.length; j += 1) {
                pillHit += boxOverlapArea(cand, placedBoxes[j], 4);
            }
            var bodyHit = 0;
            for (var k = 0; k < obstacleBoxes.length; k += 1) {
                bodyHit += boxOverlapArea(cand, obstacleBoxes[k], 2);
            }
            var score = pillHit * 20 + bodyHit * 2 + cand.nudge;
            if (score === 0) {
                return { x: cand.x, y: cand.y, w: boxW, h: boxH };
            }
            if (score < bestScore) {
                bestScore = score;
                best = cand;
            }
        }
        return { x: best.x, y: best.y, w: boxW, h: boxH };
    }

    function placePill(ctx, target, text, bg, border, fg, safeArea, placedBoxes, obstacleBoxes) {
        ctx.save();
        ctx.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        var padX = 6;
        var h = 18;
        var boxW = ctx.measureText(text).width + padX * 2;
        var rx = target.rx || 8;
        var ry = target.ry || 8;

        var candidates = [
            { x: target.cx + rx + 8, y: target.cy - h * 0.5 },
            { x: target.cx - boxW * 0.5, y: target.cy - ry - h - 8 },
            { x: target.cx - rx - boxW - 8, y: target.cy - h * 0.5 },
            { x: target.cx + rx + 8, y: target.cy + ry + 6 },
            { x: target.cx - boxW * 0.5, y: target.cy + ry + 8 },
            { x: target.cx - rx - boxW - 8, y: target.cy - ry - h - 8 },
            { x: target.cx + rx + 8, y: target.cy - ry - h - 8 }
        ];

        var chosen = chooseBestBox(candidates, boxW, h, safeArea, placedBoxes, obstacleBoxes);
        placedBoxes.push(chosen);

        drawRoundedRectPath(ctx, chosen.x, chosen.y, chosen.w, chosen.h, 4);
        ctx.fillStyle = withAlpha(bg, 0.94);
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = border;
        ctx.stroke();

        ctx.fillStyle = fg;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(text, chosen.x + padX, chosen.y + chosen.h * 0.5 + 0.5);
        ctx.restore();
    }

    function drawRoundedRectPath(ctx, x, y, w, h, r) {
        var radius = Math.min(r, w * 0.5, h * 0.5);
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + w - radius, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
        ctx.lineTo(x + w, y + h - radius);
        ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
        ctx.lineTo(x + radius, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
    }

    function drawPill(ctx, x, y, text, bg, border, fg, maxCanvasW) {
        ctx.save();
        ctx.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        var padX = 6;
        var h = 18;
        var textW = ctx.measureText(text).width;
        var boxW = textW + padX * 2;
        var drawX = clamp(x, 6, Math.max(6, maxCanvasW - boxW - 6));
        var drawY = Math.max(6, y);

        drawRoundedRectPath(ctx, drawX, drawY, boxW, h, 4);
        ctx.fillStyle = withAlpha(bg, 0.92);
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = border;
        ctx.stroke();

        ctx.fillStyle = fg;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(text, drawX + padX, drawY + h * 0.5 + 0.5);
        ctx.restore();
    }

    CampusRoadTwinCanvas.prototype.setMode = function (mode) {
        if (this._destroyed) return;
        this._mode = mode === "replay" ? "replay" : "simulation";
        this._petCells = Object.create(null);
        this._vehTrails = Object.create(null);
        this._replayPrevTracks = Object.create(null);
        this._lastReplayFrameIdx = -1;
        if (this._mode === "simulation") {
            this._simTimeS = (this._userPaused || this._motionTracker.isReduced()) ? 1.55 : 0.0;
            this._simFrameCount = Math.floor(this._simTimeS * 25 + 1e-9) + 1;
            this._lastEmittedRisk = -1;
            this._lastEmittedAlerts = -1;
            this._seedInitialPetTrail();
            this._stepSimulation(0, true);
        } else {
            this._stopLoop();
            this._renderCurrent();
        }
        this._syncLoop();
    };

    CampusRoadTwinCanvas.prototype.setScenario = function (id) {
        if (this._destroyed) return;
        this._scenarioId = id || "crosswalk_braking";
        this._simTimeS = (this._userPaused || this._motionTracker.isReduced()) ? 1.55 : 0.0;
        this._simFrameCount = Math.floor(this._simTimeS * 25 + 1e-9) + 1;
        this._lastEmittedRisk = -1;
        this._lastEmittedAlerts = -1;
        this._petCells = Object.create(null);
        this._vehTrails = Object.create(null);
        this._seedInitialPetTrail();
        if (this._mode === "simulation") {
            this._stepSimulation(0, true);
        }
    };

    CampusRoadTwinCanvas.prototype.setConfig = function (config) {
        if (this._destroyed) return;
        var cfg = config || {};
        this._config = {
            speedModel: cfg.speedModel || this._config.speedModel,
            conflictModel: cfg.conflictModel || this._config.conflictModel,
            stitching: cfg.stitching !== undefined ? Boolean(cfg.stitching) : this._config.stitching,
            cyclistVru: cfg.cyclistVru !== undefined ? Boolean(cfg.cyclistVru) : this._config.cyclistVru
        };
        this._lastEmittedRisk = -1;
        this._lastEmittedAlerts = -1;
        if (this._mode === "simulation") {
            this._seedInitialPetTrail();
            this._stepSimulation(0, true);
        }
    };

    CampusRoadTwinCanvas.prototype.setPaused = function (paused) {
        if (this._destroyed) return;
        this._userPaused = Boolean(paused);
        this._syncLoop();
        if (this._mode === "simulation") {
            this._emitSimTelemetry(true);
        }
    };

    CampusRoadTwinCanvas.prototype.isPaused = function () {
        return this._userPaused;
    };

    CampusRoadTwinCanvas.prototype.setReplayFrame = function (frameData) {
        if (this._destroyed || !frameData) return;
        var fd = frameData;
        var frameIdx = toNum(fd.frameIndex !== undefined ? fd.frameIndex : fd.frame_idx, 0);
        var ts = toNum(fd.timestampS !== undefined ? fd.timestampS : fd.timestamp_s, frameIdx * 0.1);

        if (frameIdx < this._lastReplayFrameIdx || Math.abs(frameIdx - this._lastReplayFrameIdx) > 3) {
            this._petCells = Object.create(null);
            this._vehTrails = Object.create(null);
            this._replayPrevTracks = Object.create(null);
        }
        this._lastReplayFrameIdx = frameIdx;

        var rawVehicles = Array.isArray(fd.vehicles) ? fd.vehicles : [];
        var rawVrus = Array.isArray(fd.vrus) ? fd.vrus : [];
        var rawObjects = Array.isArray(fd.objects) ? fd.objects : (Array.isArray(fd.tracks) ? fd.tracks : []);
        for (var oi = 0; oi < rawObjects.length; oi += 1) {
            var obj = rawObjects[oi];
            var cat = String(obj.category || obj.kind || "").toLowerCase();
            if (cat === "pedestrian" || cat === "person" || cat === "bicycle" || cat === "motorcycle") {
                rawVrus.push(obj);
            } else {
                rawVehicles.push(obj);
            }
        }

        var self = this;
        var normVehicles = [];
        var normVrus = [];

        function normalizeTrack(item, isVeh) {
            var tid = item.trackId !== undefined ? item.trackId : (item.track_id !== undefined ? item.track_id : item.id);
            var x = toNum(item.x !== undefined ? item.x : item.x_m, NaN);
            var y = toNum(item.y !== undefined ? item.y : item.y_m, NaN);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

            var kind = String(item.kind || item.vehicle_kind || item.target_kind || item.category || (isVeh ? "car" : "pedestrian")).toLowerCase();
            var dims = VEHICLE_DIMS[kind] || (isVeh ? VEHICLE_DIMS.car : VEHICLE_DIMS.pedestrian);
            var spdKmh = toNum(item.speedKmh !== undefined ? item.speedKmh : item.speed_kmh, 0);
            var spdMs = toNum(item.speedMs !== undefined ? item.speedMs : item.speed_ms, spdKmh / 3.6);

            var vx = toNum(item.vx !== undefined ? item.vx : item.vx_ms, NaN);
            var vy = toNum(item.vy !== undefined ? item.vy : item.vy_ms, NaN);
            var headingDeg = toNum(item.headingDeg !== undefined ? item.headingDeg : item.heading_deg, NaN);

            var prev = self._replayPrevTracks[String(tid)];
            if ((!Number.isFinite(vx) || !Number.isFinite(vy)) && prev && ts > prev.ts) {
                var dt = ts - prev.ts;
                var estVx = (x - prev.x) / dt;
                var estVy = (y - prev.y) / dt;
                var mag = Math.hypot(estVx, estVy);
                if (mag > 0.15) {
                    vx = estVx;
                    vy = estVy;
                }
            }
            if (!Number.isFinite(vx) || !Number.isFinite(vy)) {
                if (Number.isFinite(headingDeg)) {
                    var hr = headingDeg * Math.PI / 180;
                    vx = Math.cos(hr) * spdMs;
                    vy = Math.sin(hr) * spdMs;
                } else {
                    vx = 0;
                    vy = isVeh ? spdMs : 0;
                }
            }
            var headingRad = Number.isFinite(headingDeg)
                ? headingDeg * Math.PI / 180
                : (Math.hypot(vx, vy) > 0.15 ? Math.atan2(vy, vx) : (prev ? prev.headingRad : Math.PI * 0.5));

            self._replayPrevTracks[String(tid)] = { x: x, y: y, ts: ts, headingRad: headingRad };

            return {
                trackId: tid,
                kind: kind,
                category: item.category || kind,
                x: x,
                y: y,
                vx: vx,
                vy: vy,
                accelMs2: toNum(item.accelMs2 !== undefined ? item.accelMs2 : item.ax_ms2, 0),
                headingRad: headingRad,
                yawRate: toNum(item.yawRate !== undefined ? item.yawRate : item.yaw_rate_rad_s, 0),
                lengthM: toNum(item.lengthM !== undefined ? item.lengthM : item.length_m, dims.lengthM),
                widthM: toNum(item.widthM !== undefined ? item.widthM : item.width_m, dims.widthM),
                radiusM: toNum(item.radiusM, 0.42),
                speedKmh: spdKmh,
                riskLevel: parseRiskLevel(item.riskLevel !== undefined ? item.riskLevel : item.risk_level),
                inParkingZone: Boolean(item.inParkingZone || item.in_parking_zone)
            };
        }

        for (var vi = 0; vi < rawVehicles.length; vi += 1) {
            var nv = normalizeTrack(rawVehicles[vi], true);
            if (nv) normVehicles.push(nv);
        }
        for (var pi = 0; pi < rawVrus.length; pi += 1) {
            var np = normalizeTrack(rawVrus[pi], false);
            if (np) normVrus.push(np);
        }

        var rawConflicts = Array.isArray(fd.conflicts) ? fd.conflicts : [];
        var normConflicts = [];
        for (var ci = 0; ci < rawConflicts.length; ci += 1) {
            var rc = rawConflicts[ci];
            var cObj = rc.assessment ? rc.assessment : rc;
            var rLvl = parseRiskLevel(cObj.riskLevel !== undefined ? cObj.riskLevel : cObj.risk_level);
            var vehTid = cObj.vehTrackId !== undefined ? cObj.vehTrackId : cObj.veh_track_id;
            var pedTid = cObj.pedTrackId !== undefined ? cObj.pedTrackId : cObj.ped_track_id;

            var matchedVeh = findById(normVehicles, vehTid);
            if (matchedVeh && rLvl > matchedVeh.riskLevel) matchedVeh.riskLevel = rLvl;
            var matchedVru = findById(normVrus, pedTid);
            if (matchedVru && rLvl > matchedVru.riskLevel) matchedVru.riskLevel = rLvl;

            normConflicts.push({
                vehTrackId: vehTid,
                pedTrackId: pedTid,
                riskLevel: rLvl,
                ttcSeconds: toNum(cObj.ttcSeconds !== undefined ? cObj.ttcSeconds : cObj.ttc_seconds, null),
                petSeconds: toNum(cObj.petSeconds !== undefined ? cObj.petSeconds : cObj.pet_seconds, null),
                minDistanceM: toNum(cObj.minDistanceM !== undefined ? cObj.minDistanceM : cObj.min_distance_m, null),
                conflictGroundXY: cObj.conflictGroundXY || cObj.conflict_ground_xy || null
            });
        }

        var bounds = normalizeBounds(fd.bounds);
        // Auto-expand bounds cleanly if no custom bounds passed and coordinates exceed default
        if (!fd.bounds) {
            var allPts = normVehicles.concat(normVrus);
            for (var bi = 0; bi < allPts.length; bi += 1) {
                if (allPts[bi].x - 1.0 < bounds.xMin) bounds.xMin = Math.floor(allPts[bi].x - 1.0);
                if (allPts[bi].x + 1.0 > bounds.xMax) bounds.xMax = Math.ceil(allPts[bi].x + 1.0);
                if (allPts[bi].y - 1.5 < bounds.yMin) bounds.yMin = Math.floor(allPts[bi].y - 1.5);
                if (allPts[bi].y + 1.5 > bounds.yMax) bounds.yMax = Math.ceil(allPts[bi].y + 1.5);
            }
        }

        this._recordVehTrails(normVehicles, ts);
        this._recordPetGrid(normVehicles, normVrus, ts);

        var maxRisk = parseRiskLevel(fd.riskLevel !== undefined ? fd.riskLevel : fd.risk_level);
        for (var rvi = 0; rvi < normVehicles.length; rvi += 1) {
            if (normVehicles[rvi].riskLevel > maxRisk) maxRisk = normVehicles[rvi].riskLevel;
        }
        for (var rui = 0; rui < normVrus.length; rui += 1) {
            if (normVrus[rui].riskLevel > maxRisk) maxRisk = normVrus[rui].riskLevel;
        }
        for (var rci = 0; rci < normConflicts.length; rci += 1) {
            if (normConflicts[rci].riskLevel > maxRisk) maxRisk = normConflicts[rci].riskLevel;
        }

        this._currentRenderState = {
            bounds: bounds,
            parkingZones: Array.isArray(fd.parkingZones) ? fd.parkingZones : [],
            vehicles: normVehicles,
            vrus: normVrus,
            conflicts: normConflicts,
            label: fd.label || ("REPLAY · F" + padId(frameIdx) + " · " + ts.toFixed(1) + "s"),
            cameraLabel: fd.cameraLabel || fd.camera_label || null,
            cameraShortLabel: fd.cameraShortLabel || fd.camera_short_label || null,
            simTimeS: ts,
            maxRisk: maxRisk,
            speedModel: (this._config && this._config.speedModel) || "ctrv",
            conflictModel: (this._config && this._config.conflictModel) || "obb_ssm"
        };
        this._measureCanvas();
        this._renderCurrent();
    };

    CampusRoadTwinCanvas.prototype.refreshTheme = function () {
        if (this._destroyed) return;
        this._palette = readThemePalette(this._probe);
        this._measureCanvas();
        this._renderCurrent();
    };

    CampusRoadTwinCanvas.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        this._stopLoop();
        detachResizeObserver(this);
        if (this._intersectionObserver) {
            this._intersectionObserver.disconnect();
            this._intersectionObserver = null;
        }
        this._unobserveTheme();
        this._motionTracker.destroy();
        if (this._probe && this._probe.parentNode) {
            this._probe.parentNode.removeChild(this._probe);
        }
    };

    /* ======================================================================
     * 4. SpeedTimelineChart (Dual-Band Speed & Conflict Risk Chart)
     * ====================================================================== */
    function SpeedTimelineChart(canvasEl, options) {
        if (!(this instanceof SpeedTimelineChart)) {
            return new SpeedTimelineChart(canvasEl, options);
        }
        if (!canvasEl) {
            throw new Error("SpeedTimelineChart requires a valid <canvas> element");
        }
        var opts = options || {};
        this._canvas = canvasEl;
        this._ctx = canvasEl.getContext("2d");
        this._speedLimitKmh = toNum(opts.speedLimitKmh, 30.0);
        this._onHover = typeof opts.onHover === "function" ? opts.onHover : null;
        this._onSelectFrame = typeof opts.onSelectFrame === "function" ? opts.onSelectFrame : null;
        this._points = [];
        this._activeFrameIndex = null;
        this._hoverIdx = -1;
        this._destroyed = false;

        this._cssWidth = 0;
        this._cssHeight = 0;
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        this._plotBox = null;

        this._probe = createColorProbe();
        this._palette = readThemePalette(this._probe);

        var self = this;
        this._unobserveTheme = observeThemeChanges(function () {
            self.refreshTheme();
        });

        this._onPointerMove = function (ev) {
            self._handlePointerMove(ev);
        };
        this._onPointerLeave = function () {
            if (self._hoverIdx !== -1) {
                self._hoverIdx = -1;
                if (self._onHover) self._onHover(null);
                self._render();
            }
        };
        this._onClick = function (ev) {
            self._handleClick(ev);
        };

        this._canvas.addEventListener("pointermove", this._onPointerMove);
        this._canvas.addEventListener("pointerleave", this._onPointerLeave);
        this._canvas.addEventListener("click", this._onClick);

        attachResizeObserver(this._canvas, this, function () {
            self._measure();
            self._render();
        });

        this._measure();
        this._render();
    }

    SpeedTimelineChart.prototype._measure = function () {
        var rect = this._canvas.getBoundingClientRect();
        var w = Math.max(0, rect.width || this._canvas.clientWidth || 0);
        var h = Math.max(0, rect.height || this._canvas.clientHeight || 0);
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        this._cssWidth = w;
        this._cssHeight = h;
        if (w >= 2 && h >= 2) {
            var pw = Math.round(w * this._dpr);
            var ph = Math.round(h * this._dpr);
            if (this._canvas.width !== pw || this._canvas.height !== ph) {
                this._canvas.width = pw;
                this._canvas.height = ph;
            }
        }
    };

    SpeedTimelineChart.prototype._indexFromPointer = function (ev) {
        if (!this._plotBox || this._points.length === 0) return -1;
        var rect = this._canvas.getBoundingClientRect();
        var x = ev.clientX - rect.left;
        var box = this._plotBox;
        if (x < box.x - 10 || x > box.x + box.w + 10) return -1;
        if (this._points.length === 1) return 0;
        var ratio = clamp((x - box.x) / Math.max(1, box.w), 0, 1);
        return Math.round(ratio * (this._points.length - 1));
    };

    SpeedTimelineChart.prototype._handlePointerMove = function (ev) {
        var idx = this._indexFromPointer(ev);
        if (idx !== this._hoverIdx) {
            this._hoverIdx = idx;
            if (this._onHover) {
                this._onHover(idx >= 0 ? this._points[idx] : null);
            }
            this._render();
        }
    };

    SpeedTimelineChart.prototype._handleClick = function (ev) {
        var idx = this._indexFromPointer(ev);
        if (idx >= 0 && this._points[idx]) {
            this._activeFrameIndex = this._points[idx].frameIndex;
            this._render();
            if (this._onSelectFrame) {
                this._onSelectFrame(this._points[idx].frameIndex);
            }
        }
    };

    SpeedTimelineChart.prototype._render = function () {
        if (!this._ctx || this._cssWidth < 2 || this._cssHeight < 2) return;
        var ctx = this._ctx;
        var pal = this._palette;
        var w = this._cssWidth;
        var h = this._cssHeight;

        ctx.save();
        ctx.scale(this._dpr, this._dpr);
        ctx.clearRect(0, 0, w, h);

        ctx.fillStyle = pal.bgCard;
        ctx.fillRect(0, 0, w, h);

        var padL = 44;
        var padR = 16;
        var padT = 18;
        var padB = 24;
        var plotW = Math.max(30, w - padL - padR);
        var plotH = Math.max(40, h - padT - padB);
        this._plotBox = { x: padL, y: padT, w: plotW, h: plotH };

        if (this._points.length === 0) {
            ctx.fillStyle = pal.textMuted;
            ctx.font = "400 12px system-ui, -apple-system, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("暂无时序轨迹数据", w * 0.5, h * 0.5);
            ctx.restore();
            return;
        }

        // Dual band geometry: top 64% speed line, bottom 28% conflict/TTC markers
        var topBandH = plotH * 0.64;
        var gapH = plotH * 0.08;
        var botBandY = padT + topBandH + gapH;
        var botBandH = plotH - topBandH - gapH;

        var maxSpd = this._speedLimitKmh * 1.25;
        for (var i = 0; i < this._points.length; i += 1) {
            if (this._points[i].speedKmh > maxSpd) maxSpd = this._points[i].speedKmh * 1.15;
        }
        maxSpd = Math.max(10, Math.ceil(maxSpd / 5) * 5);

        function xAt(idx, total) {
            if (total <= 1) return padL + plotW * 0.5;
            return padL + (idx / (total - 1)) * plotW;
        }
        function ySpeed(spd) {
            return padT + topBandH - clamp(spd / maxSpd, 0, 1) * topBandH;
        }

        // Horizontal Y ticks & speed limit dashed hairline
        ctx.font = "500 10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        var ticks = [0, Math.round(maxSpd * 0.5), maxSpd];
        for (var ti = 0; ti < ticks.length; ti += 1) {
            var ty = ySpeed(ticks[ti]);
            ctx.strokeStyle = pal.gridLine;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(padL, ty);
            ctx.lineTo(padL + plotW, ty);
            ctx.stroke();

            ctx.fillStyle = pal.textMuted;
            ctx.textAlign = "right";
            ctx.textBaseline = "middle";
            ctx.fillText(String(ticks[ti]), padL - 6, ty);
        }

        var limitY = ySpeed(this._speedLimitKmh);
        if (limitY >= padT - 4 && limitY <= padT + topBandH + 4) {
            ctx.save();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = withAlpha(pal.accent, 0.78);
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(padL, limitY);
            ctx.lineTo(padL + plotW, limitY);
            ctx.stroke();
            ctx.restore();

            ctx.fillStyle = pal.accent;
            ctx.textAlign = "right";
            ctx.textBaseline = "bottom";
            ctx.fillText(this._speedLimitKmh.toFixed(0) + " km/h", padL + plotW, limitY - 3);
        }

        // Band divider hairline
        var divY = padT + topBandH + gapH * 0.5;
        ctx.strokeStyle = pal.hairline;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, divY);
        ctx.lineTo(padL + plotW, divY);
        ctx.stroke();

        // Speed area fill & stroke
        var n = this._points.length;
        ctx.save();
        ctx.beginPath();
        for (var j = 0; j < n; j += 1) {
            var px = xAt(j, n);
            var py = ySpeed(this._points[j].speedKmh);
            if (j === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.strokeStyle = pal.textPrimary;
        ctx.lineWidth = 1.8;
        ctx.stroke();

        ctx.lineTo(xAt(n - 1, n), padT + topBandH);
        ctx.lineTo(xAt(0, n), padT + topBandH);
        ctx.closePath();
        var grad = ctx.createLinearGradient(0, padT, 0, padT + topBandH);
        grad.addColorStop(0, withAlpha(pal.textPrimary, 0.14));
        grad.addColorStop(1, withAlpha(pal.textPrimary, 0.0));
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.restore();

        // Bottom band: risk-colored markers + TTC severity stems
        ctx.fillStyle = pal.textMuted;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText("风险", padL - 6, botBandY + botBandH * 0.5);

        for (var m = 0; m < n; m += 1) {
            var pt = this._points[m];
            var mx = xAt(m, n);
            var rLvl = pt.riskLevel;
            var rColor = rLvl === 0 ? withAlpha(pal.textMuted, 0.35) : riskColor(pal, rLvl);

            var stemRatio = 0.18;
            if (rLvl >= 2) {
                stemRatio = pt.ttcS !== null ? clamp(1 - pt.ttcS / 4.0, 0.4, 0.95) : (rLvl === 3 ? 0.9 : 0.6);
            } else if (rLvl === 1) {
                stemRatio = 0.42;
            }
            var stemH = Math.max(3, botBandH * stemRatio);
            ctx.strokeStyle = rColor;
            ctx.lineWidth = rLvl >= 2 ? 2.6 : 1.5;
            ctx.beginPath();
            ctx.moveTo(mx, botBandY + botBandH);
            ctx.lineTo(mx, botBandY + botBandH - stemH);
            ctx.stroke();

            if (rLvl >= 2) {
                ctx.fillStyle = rColor;
                ctx.beginPath();
                ctx.arc(mx, botBandY + botBandH - stemH, 2.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // X axis time ticks
        var xTickIndices = n <= 2 ? [0, n - 1] : [0, Math.floor((n - 1) * 0.5), n - 1];
        for (var xi = 0; xi < xTickIndices.length; xi += 1) {
            var idx = xTickIndices[xi];
            if (idx < 0 || idx >= n) continue;
            var tx = xAt(idx, n);
            ctx.fillStyle = pal.textMuted;
            ctx.textAlign = idx === 0 ? "left" : (idx === n - 1 ? "right" : "center");
            ctx.textBaseline = "top";
            ctx.fillText(this._points[idx].t.toFixed(1) + "s", tx, padT + plotH + 6);
        }

        // Active frame cursor
        if (this._activeFrameIndex !== null) {
            for (var ai = 0; ai < n; ai += 1) {
                if (this._points[ai].frameIndex === this._activeFrameIndex) {
                    var ax = xAt(ai, n);
                    ctx.strokeStyle = pal.accent;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.moveTo(ax, padT);
                    ctx.lineTo(ax, padT + plotH);
                    ctx.stroke();

                    ctx.fillStyle = pal.accent;
                    ctx.beginPath();
                    ctx.arc(ax, ySpeed(this._points[ai].speedKmh), 3.8, 0, Math.PI * 2);
                    ctx.fill();
                    break;
                }
            }
        }

        // Hover crosshair & compact telemetry readout
        if (this._hoverIdx >= 0 && this._hoverIdx < n) {
            var hp = this._points[this._hoverIdx];
            var hx = xAt(this._hoverIdx, n);
            ctx.save();
            ctx.setLineDash([3, 3]);
            ctx.strokeStyle = pal.textSecondary;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(hx, padT);
            ctx.lineTo(hx, padT + plotH);
            ctx.stroke();
            ctx.restore();

            ctx.fillStyle = pal.bgCard;
            ctx.strokeStyle = pal.textPrimary;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(hx, ySpeed(hp.speedKmh), 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            var ttcLabel = hp.ttcS !== null ? (" · TTC " + hp.ttcS.toFixed(2) + "s") : "";
            var tipText = "F" + hp.frameIndex + " · " + hp.t.toFixed(1) + "s · " + hp.speedKmh.toFixed(1) + "km/h" + ttcLabel + " · L" + hp.riskLevel;
            ctx.save();
            ctx.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
            var tipW = ctx.measureText(tipText).width + 12;
            ctx.restore();
            var tipX = (hx + 8 + tipW > w - 8) ? Math.max(8, hx - tipW - 8) : (hx + 8);
            drawPill(ctx, tipX, padT + 2, tipText, pal.bgRaised, pal.hairline, pal.textPrimary, w);
        }

        ctx.restore();
    };

    SpeedTimelineChart.prototype.setData = function (points, speedLimitKmh) {
        if (this._destroyed) return;
        if (speedLimitKmh !== undefined) {
            this._speedLimitKmh = toNum(speedLimitKmh, this._speedLimitKmh);
        }
        var raw = Array.isArray(points) ? points : [];
        var normalized = [];
        for (var i = 0; i < raw.length; i += 1) {
            var p = raw[i];
            if (!p) continue;
            var frameIndex = toNum(p.frameIndex !== undefined ? p.frameIndex : p.frame_idx, i);
            var t = toNum(p.t !== undefined ? p.t : (p.timestampS !== undefined ? p.timestampS : p.timestamp_s), frameIndex * 0.1);
            var speedKmh = toNum(p.speedKmh !== undefined ? p.speedKmh : p.speed_kmh, 0);
            var ttcS = toNum(p.ttcS !== undefined ? p.ttcS : (p.ttc_s !== undefined ? p.ttc_s : p.ttc_seconds), null);
            var riskLevel = parseRiskLevel(p.riskLevel !== undefined ? p.riskLevel : p.risk_level);
            normalized.push({
                frameIndex: frameIndex,
                t: t,
                speedKmh: speedKmh,
                ttcS: ttcS,
                riskLevel: riskLevel
            });
        }
        this._points = normalized;
        this._hoverIdx = -1;
        this._measure();
        this._render();
    };

    SpeedTimelineChart.prototype.setActiveFrame = function (frameIndex) {
        if (this._destroyed) return;
        this._activeFrameIndex = frameIndex === null || frameIndex === undefined ? null : toNum(frameIndex, null);
        this._measure();
        this._render();
    };

    SpeedTimelineChart.prototype.refreshTheme = function () {
        if (this._destroyed) return;
        this._palette = readThemePalette(this._probe);
        this._measure();
        this._render();
    };

    SpeedTimelineChart.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        this._canvas.removeEventListener("pointermove", this._onPointerMove);
        this._canvas.removeEventListener("pointerleave", this._onPointerLeave);
        this._canvas.removeEventListener("click", this._onClick);
        detachResizeObserver(this);
        this._unobserveTheme();
        if (this._probe && this._probe.parentNode) {
            this._probe.parentNode.removeChild(this._probe);
        }
    };

    /* ======================================================================
     * 5. CalibrationCanvas (Interactive Homography & Parking Zone Editor)
     * ====================================================================== */
    function roundCoord1(val) {
        return Math.round(val * 10) / 10;
    }

    function clonePointPair(pt) {
        if (!Array.isArray(pt) || pt.length < 2) return null;
        var u = toNum(pt[0], NaN);
        var v = toNum(pt[1], NaN);
        if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
        return [u, v];
    }

    function clonePointList(list) {
        var out = [];
        if (!Array.isArray(list)) return out;
        for (var i = 0; i < list.length; i += 1) {
            var pair = clonePointPair(list[i]);
            if (pair) out.push(pair);
        }
        return out;
    }

    function clonePolygonList(polys) {
        var out = [];
        if (!Array.isArray(polys)) return out;
        for (var i = 0; i < polys.length; i += 1) {
            var poly = clonePointList(polys[i]);
            if (poly.length >= 3) out.push(poly);
        }
        return out;
    }

    function CalibrationCanvas(canvasEl, options) {
        if (!(this instanceof CalibrationCanvas)) {
            return new CalibrationCanvas(canvasEl, options);
        }
        if (!canvasEl) {
            throw new Error("CalibrationCanvas requires a valid <canvas> element");
        }
        var opts = options || {};
        this._canvas = canvasEl;
        this._ctx = canvasEl.getContext("2d");
        this._onChange = typeof opts.onChange === "function" ? opts.onChange : null;

        this._tool = "points";
        this._image = null;
        this._imageReady = false;
        this._imageWidth = 0;
        this._imageHeight = 0;
        this._loadToken = 0;

        this._imagePoints = [];
        this._parkingZonesImage = [];
        this._currentParkingDraft = [];

        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._hoverPos = null;
        this._lastEmitTs = 0;
        this._emitTimer = null;
        this._destroyed = false;

        this._cssWidth = 0;
        this._cssHeight = 0;
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);

        this._probe = createColorProbe();
        this._palette = readThemePalette(this._probe);

        if (this._canvas.style && !this._canvas.style.touchAction) {
            this._canvas.style.touchAction = "none";
        }

        var self = this;
        this._motionTracker = createReducedMotionTracker(function () {
            if (!self._destroyed) {
                self._render();
            }
        });
        this._unobserveTheme = observeThemeChanges(function () {
            self.refreshTheme();
        });

        this._onPointerDown = function (ev) {
            self._handlePointerDown(ev);
        };
        this._onPointerMove = function (ev) {
            self._handlePointerMove(ev);
        };
        this._onPointerUp = function (ev) {
            self._handlePointerUp(ev);
        };
        this._onPointerCancel = function (ev) {
            self._handlePointerUp(ev);
        };
        this._onPointerLeave = function () {
            if (!self._dragState && (self._hoverVertexIdx !== -1 || self._hoverPos !== null)) {
                self._hoverVertexIdx = -1;
                self._hoverPos = null;
                self._updateCursor();
                self._render();
            }
        };

        this._canvas.addEventListener("pointerdown", this._onPointerDown);
        this._canvas.addEventListener("pointermove", this._onPointerMove);
        this._canvas.addEventListener("pointerup", this._onPointerUp);
        this._canvas.addEventListener("pointercancel", this._onPointerCancel);
        this._canvas.addEventListener("pointerleave", this._onPointerLeave);

        attachResizeObserver(this._canvas, this, function () {
            self._measure();
            self._render();
        });

        this._measure();
        this._updateCursor();
        this._render();
    }

    CalibrationCanvas.prototype._measure = function () {
        var rect = this._canvas.getBoundingClientRect();
        var w = Math.max(0, rect.width || this._canvas.clientWidth || 0);
        var h = Math.max(0, rect.height || this._canvas.clientHeight || 0);
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        this._cssWidth = w;
        this._cssHeight = h;
        if (w >= 2 && h >= 2) {
            var pw = Math.round(w * this._dpr);
            var ph = Math.round(h * this._dpr);
            if (this._canvas.width !== pw || this._canvas.height !== ph) {
                this._canvas.width = pw;
                this._canvas.height = ph;
            }
        }
    };

    CalibrationCanvas.prototype._hasFrameDimensions = function () {
        return this._imageWidth > 0 && this._imageHeight > 0;
    };

    CalibrationCanvas.prototype._computeFitRect = function () {
        var w = this._cssWidth > 2
            ? this._cssWidth
            : (this._canvas.clientWidth || (this._canvas.width / (this._dpr || 1)) || 640);
        var h = this._cssHeight > 2
            ? this._cssHeight
            : (this._canvas.clientHeight || (this._canvas.height / (this._dpr || 1)) || 360);
        var pad = clamp(Math.floor(Math.min(w, h) * 0.045), 10, 18);
        var availW = Math.max(20, w - pad * 2);
        var availH = Math.max(20, h - pad * 2);
        var imgW = this._imageWidth > 0 ? this._imageWidth : 1280;
        var imgH = this._imageHeight > 0 ? this._imageHeight : 720;
        var scale = Math.min(availW / imgW, availH / imgH);
        var drawW = imgW * scale;
        var drawH = imgH * scale;
        return {
            x: pad + (availW - drawW) * 0.5,
            y: pad + (availH - drawH) * 0.5,
            w: drawW,
            h: drawH,
            scale: scale,
            imgW: imgW,
            imgH: imgH
        };
    };

    CalibrationCanvas.prototype.imageToCanvas = function (u, v) {
        var imgU = u;
        var imgV = v;
        if (Array.isArray(u)) {
            imgU = u[0];
            imgV = u[1];
        } else if (u && typeof u === "object") {
            imgU = u.u !== undefined ? u.u : u.x;
            imgV = u.v !== undefined ? u.v : u.y;
        }
        var fit = this._computeFitRect();
        return {
            x: fit.x + toNum(imgU, 0) * fit.scale,
            y: fit.y + toNum(imgV, 0) * fit.scale
        };
    };

    CalibrationCanvas.prototype.canvasToImage = function (cx, cy) {
        var px = cx;
        var py = cy;
        if (Array.isArray(cx)) {
            px = cx[0];
            py = cx[1];
        } else if (cx && typeof cx === "object") {
            px = cx.x;
            py = cx.y;
        }
        var fit = this._computeFitRect();
        var rawU = (toNum(px, fit.x) - fit.x) / Math.max(1e-6, fit.scale);
        var rawV = (toNum(py, fit.y) - fit.y) / Math.max(1e-6, fit.scale);
        return {
            u: clamp(roundCoord1(rawU), 0, fit.imgW),
            v: clamp(roundCoord1(rawV), 0, fit.imgH)
        };
    };

    CalibrationCanvas.prototype._pointerToCanvas = function (ev) {
        var rect = this._canvas.getBoundingClientRect();
        var x = ev.clientX - (rect.left || 0);
        var y = ev.clientY - (rect.top || 0);
        if (rect.width > 1 && this._cssWidth > 1 && Math.abs(rect.width - this._cssWidth) > 1) {
            x = x * (this._cssWidth / rect.width);
        }
        if (rect.height > 1 && this._cssHeight > 1 && Math.abs(rect.height - this._cssHeight) > 1) {
            y = y * (this._cssHeight / rect.height);
        }
        return { x: x, y: y };
    };

    CalibrationCanvas.prototype._activeVertexList = function () {
        return this._tool === "parking" ? this._currentParkingDraft : this._imagePoints;
    };

    CalibrationCanvas.prototype._findNearestVertex = function (cx, cy, list, maxDistPx) {
        var threshold = maxDistPx !== undefined ? maxDistPx : 14;
        var bestIdx = -1;
        var bestDist = Infinity;
        for (var i = 0; i < list.length; i += 1) {
            var pt = this.imageToCanvas(list[i][0], list[i][1]);
            var d = Math.hypot(pt.x - cx, pt.y - cy);
            if (d <= threshold && d < bestDist) {
                bestDist = d;
                bestIdx = i;
            }
        }
        return bestIdx;
    };

    CalibrationCanvas.prototype._updateCursor = function () {
        if (!this._canvas || !this._canvas.style) return;
        if (!this._hasFrameDimensions()) {
            this._canvas.style.cursor = "default";
        } else if (this._dragState) {
            this._canvas.style.cursor = "grabbing";
        } else if (this._hoverVertexIdx >= 0) {
            this._canvas.style.cursor = "grab";
        } else {
            this._canvas.style.cursor = "crosshair";
        }
    };

    CalibrationCanvas.prototype._emitChange = function (immediate) {
        if (this._destroyed || !this._onChange) return;
        var now = typeof performance !== "undefined" ? performance.now() : Date.now();
        if (immediate) {
            if (this._emitTimer !== null && typeof global.clearTimeout === "function") {
                global.clearTimeout(this._emitTimer);
                this._emitTimer = null;
            }
            this._lastEmitTs = now;
            this._onChange(this.getState());
            return;
        }
        var elapsed = now - this._lastEmitTs;
        if (elapsed >= 48) {
            if (this._emitTimer !== null && typeof global.clearTimeout === "function") {
                global.clearTimeout(this._emitTimer);
                this._emitTimer = null;
            }
            this._lastEmitTs = now;
            this._onChange(this.getState());
        } else if (this._emitTimer === null && typeof global.setTimeout === "function") {
            var self = this;
            var wait = Math.max(10, 48 - elapsed);
            this._emitTimer = global.setTimeout(function () {
                self._emitTimer = null;
                if (self._destroyed || !self._onChange) return;
                self._lastEmitTs = typeof performance !== "undefined" ? performance.now() : Date.now();
                self._onChange(self.getState());
            }, wait);
        }
    };

    CalibrationCanvas.prototype._handlePointerDown = function (ev) {
        if (this._destroyed) return;
        if (ev.button !== undefined && ev.button !== 0) return;
        if (!this._hasFrameDimensions()) return;

        var pos = this._pointerToCanvas(ev);
        this._hoverPos = pos;
        var list = this._activeVertexList();
        var hitIdx = this._findNearestVertex(pos.x, pos.y, list, 14);

        if (hitIdx >= 0) {
            this._dragState = {
                target: this._tool,
                index: hitIdx,
                pointerId: ev.pointerId,
                moved: false
            };
            this._hoverVertexIdx = hitIdx;
            if (ev.pointerId !== undefined && typeof this._canvas.setPointerCapture === "function") {
                try {
                    this._canvas.setPointerCapture(ev.pointerId);
                } catch (_err) {
                    // Ignore synthetic pointer capture errors
                }
            }
            this._updateCursor();
            this._render();
            return;
        }

        var fit = this._computeFitRect();
        var margin = 12;
        if (pos.x < fit.x - margin || pos.x > fit.x + fit.w + margin ||
            pos.y < fit.y - margin || pos.y > fit.y + fit.h + margin) {
            return;
        }

        var imgPt = this.canvasToImage(pos.x, pos.y);
        list.push([imgPt.u, imgPt.v]);
        this._hoverVertexIdx = list.length - 1;
        this._updateCursor();
        this._render();
        this._emitChange(true);
    };

    CalibrationCanvas.prototype._handlePointerMove = function (ev) {
        if (this._destroyed) return;
        if (!this._hasFrameDimensions()) return;

        var pos = this._pointerToCanvas(ev);
        this._hoverPos = pos;

        if (this._dragState) {
            if (ev.pointerId !== undefined &&
                this._dragState.pointerId !== undefined &&
                ev.pointerId !== this._dragState.pointerId) {
                return;
            }
            var list = this._dragState.target === "parking"
                ? this._currentParkingDraft
                : this._imagePoints;
            var idx = this._dragState.index;
            if (idx >= 0 && idx < list.length) {
                var imgPt = this.canvasToImage(pos.x, pos.y);
                var prev = list[idx];
                if (prev[0] !== imgPt.u || prev[1] !== imgPt.v) {
                    list[idx] = [imgPt.u, imgPt.v];
                    this._dragState.moved = true;
                    this._render();
                    this._emitChange(false);
                }
            }
            return;
        }

        var activeList = this._activeVertexList();
        var hitIdx = this._findNearestVertex(pos.x, pos.y, activeList, 14);
        var changed = hitIdx !== this._hoverVertexIdx;
        this._hoverVertexIdx = hitIdx;
        this._updateCursor();
        if (changed || activeList.length > 0) {
            this._render();
        }
    };

    CalibrationCanvas.prototype._handlePointerUp = function (ev) {
        if (this._destroyed || !this._dragState) return;
        if (ev.pointerId !== undefined &&
            this._dragState.pointerId !== undefined &&
            ev.pointerId !== this._dragState.pointerId) {
            return;
        }

        var wasMoved = this._dragState.moved;
        var ptrId = this._dragState.pointerId;
        this._dragState = null;

        if (ptrId !== undefined &&
            typeof this._canvas.hasPointerCapture === "function" &&
            typeof this._canvas.releasePointerCapture === "function") {
            try {
                if (this._canvas.hasPointerCapture(ptrId)) {
                    this._canvas.releasePointerCapture(ptrId);
                }
            } catch (_err) {
                // Ignore synthetic pointer release errors
            }
        }

        this._updateCursor();
        this._render();
        if (wasMoved || this._emitTimer !== null) {
            this._emitChange(true);
        }
    };

    CalibrationCanvas.prototype._drawCenteredBadge = function (ctx, cx, cy, text, bg, border, fg, canvasW, canvasH) {
        ctx.save();
        ctx.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        var padX = 6;
        var h = 18;
        var boxW = ctx.measureText(text).width + padX * 2;
        var x = clamp(cx - boxW * 0.5, 6, Math.max(6, canvasW - boxW - 6));
        var y = clamp(cy - h * 0.5, 6, Math.max(6, canvasH - h - 6));
        drawRoundedRectPath(ctx, x, y, boxW, h, 4);
        ctx.fillStyle = withAlpha(bg, 0.92);
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = border;
        ctx.stroke();
        ctx.fillStyle = fg;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(text, x + padX, y + h * 0.5 + 0.5);
        ctx.restore();
    };

    CalibrationCanvas.prototype._render = function () {
        if (!this._ctx || this._cssWidth < 2 || this._cssHeight < 2) return;
        var ctx = this._ctx;
        var pal = this._palette;
        var w = this._cssWidth;
        var h = this._cssHeight;

        ctx.save();
        ctx.scale(this._dpr, this._dpr);
        ctx.clearRect(0, 0, w, h);

        // Backdrop & subtle engineering dot pattern
        ctx.fillStyle = pal.roadShoulder;
        ctx.fillRect(0, 0, w, h);

        ctx.fillStyle = withAlpha(pal.textMuted, pal.isLight ? 0.14 : 0.12);
        var dotStep = 24;
        for (var dx = 12; dx < w; dx += dotStep) {
            for (var dy = 12; dy < h; dy += dotStep) {
                ctx.fillRect(dx, dy, 1, 1);
            }
        }

        var fit = this._computeFitRect();

        if (this._imageReady && this._image) {
            ctx.drawImage(this._image, fit.x, fit.y, fit.w, fit.h);
            ctx.strokeStyle = pal.hairline;
            ctx.lineWidth = 1;
            ctx.strokeRect(fit.x, fit.y, fit.w, fit.h);
        } else {
            // Crisp instrumented guide grid when no frame image is rendered
            ctx.fillStyle = pal.bgCard;
            ctx.fillRect(fit.x, fit.y, fit.w, fit.h);
            ctx.strokeStyle = pal.hairline;
            ctx.lineWidth = 1;
            ctx.strokeRect(fit.x, fit.y, fit.w, fit.h);

            var cols = 12;
            var rows = 8;
            ctx.strokeStyle = pal.gridLine;
            ctx.lineWidth = 1;
            for (var ci = 1; ci < cols; ci += 1) {
                var gx = fit.x + (fit.w * ci) / cols;
                ctx.beginPath();
                ctx.moveTo(gx, fit.y);
                ctx.lineTo(gx, fit.y + fit.h);
                ctx.stroke();
            }
            for (var ri = 1; ri < rows; ri += 1) {
                var gy = fit.y + (fit.h * ri) / rows;
                ctx.beginPath();
                ctx.moveTo(fit.x, gy);
                ctx.lineTo(fit.x + fit.w, gy);
                ctx.stroke();
            }

            // Center crosshair & corner calibration ticks
            var midX = fit.x + fit.w * 0.5;
            var midY = fit.y + fit.h * 0.5;
            ctx.strokeStyle = withAlpha(pal.roadMark, 0.85);
            ctx.beginPath();
            ctx.moveTo(midX - 24, midY);
            ctx.lineTo(midX + 24, midY);
            ctx.moveTo(midX, midY - 24);
            ctx.lineTo(midX, midY + 24);
            ctx.stroke();

            ctx.strokeStyle = withAlpha(pal.accent, 0.40);
            ctx.beginPath();
            ctx.arc(midX, midY, 16, 0, Math.PI * 2);
            ctx.stroke();

            var arm = 14;
            ctx.strokeStyle = withAlpha(pal.accent, 0.65);
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            // Top-left
            ctx.moveTo(fit.x, fit.y + arm);
            ctx.lineTo(fit.x, fit.y);
            ctx.lineTo(fit.x + arm, fit.y);
            // Top-right
            ctx.moveTo(fit.x + fit.w - arm, fit.y);
            ctx.lineTo(fit.x + fit.w, fit.y);
            ctx.lineTo(fit.x + fit.w, fit.y + arm);
            // Bottom-left
            ctx.moveTo(fit.x, fit.y + fit.h - arm);
            ctx.lineTo(fit.x, fit.y + fit.h);
            ctx.lineTo(fit.x + arm, fit.y + fit.h);
            // Bottom-right
            ctx.moveTo(fit.x + fit.w - arm, fit.y + fit.h);
            ctx.lineTo(fit.x + fit.w, fit.y + fit.h);
            ctx.lineTo(fit.x + fit.w, fit.y + fit.h - arm);
            ctx.stroke();

            if (!this._hasFrameDimensions()) {
                ctx.fillStyle = pal.textSecondary;
                ctx.font = "500 13px system-ui, -apple-system, sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("等待加载视频标定帧", midX, midY - 12);

                ctx.fillStyle = pal.textMuted;
                ctx.font = "400 11px system-ui, -apple-system, sans-serif";
                ctx.fillText("选择视频源并提取标定帧后，可点击设置 P1..PN 参考点与禁停区", midX, midY + 12);
            }

            ctx.fillStyle = pal.textMuted;
            ctx.font = "500 10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
            ctx.textAlign = "right";
            ctx.textBaseline = "bottom";
            var specTag = this._hasFrameDimensions()
                ? (this._imageWidth + "x" + this._imageHeight + " px")
                : "CALIBRATION GRID · 16:9";
            ctx.fillText(specTag, fit.x + fit.w - 10, fit.y + fit.h - 8);
        }

        // 1. Committed roadside parking exclusion zones
        for (var zi = 0; zi < this._parkingZonesImage.length; zi += 1) {
            var poly = this._parkingZonesImage[zi];
            if (!Array.isArray(poly) || poly.length < 3) continue;
            var sumX = 0;
            var sumY = 0;
            var zPts = [];
            for (var zp = 0; zp < poly.length; zp += 1) {
                var cPt = this.imageToCanvas(poly[zp][0], poly[zp][1]);
                zPts.push(cPt);
                sumX += cPt.x;
                sumY += cPt.y;
            }
            ctx.save();
            ctx.beginPath();
            for (var k = 0; k < zPts.length; k += 1) {
                if (k === 0) ctx.moveTo(zPts[k].x, zPts[k].y);
                else ctx.lineTo(zPts[k].x, zPts[k].y);
            }
            ctx.closePath();
            ctx.fillStyle = withAlpha(pal.parkingZone, 0.18);
            ctx.fill();
            ctx.setLineDash([6, 4]);
            ctx.strokeStyle = withAlpha(pal.parkingZone, 0.92);
            ctx.lineWidth = 1.8;
            ctx.stroke();
            ctx.restore();

            for (var zv = 0; zv < zPts.length; zv += 1) {
                ctx.fillStyle = pal.bgCard;
                ctx.strokeStyle = pal.parkingZone;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(zPts[zv].x, zPts[zv].y, 3.2, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }

            var zoneLabel = this._parkingZonesImage.length > 1
                ? ("禁停标定区 #" + (zi + 1))
                : "禁停标定区";
            this._drawCenteredBadge(
                ctx,
                sumX / zPts.length,
                sumY / zPts.length,
                zoneLabel,
                pal.bgCard,
                withAlpha(pal.parkingZone, 0.78),
                pal.parkingZone,
                w,
                h
            );
        }

        // 2. Active roadside parking exclusion draft polygon
        if (this._currentParkingDraft.length > 0) {
            var dPts = [];
            var dSumX = 0;
            var dSumY = 0;
            for (var di = 0; di < this._currentParkingDraft.length; di += 1) {
                var dPt = this.imageToCanvas(this._currentParkingDraft[di][0], this._currentParkingDraft[di][1]);
                dPts.push(dPt);
                dSumX += dPt.x;
                dSumY += dPt.y;
            }

            if (dPts.length >= 3) {
                ctx.save();
                ctx.beginPath();
                for (var df = 0; df < dPts.length; df += 1) {
                    if (df === 0) ctx.moveTo(dPts[df].x, dPts[df].y);
                    else ctx.lineTo(dPts[df].x, dPts[df].y);
                }
                ctx.closePath();
                ctx.fillStyle = withAlpha(pal.accent, 0.14);
                ctx.fill();
                ctx.setLineDash([3, 4]);
                ctx.strokeStyle = withAlpha(pal.accent, 0.55);
                ctx.lineWidth = 1.3;
                ctx.beginPath();
                ctx.moveTo(dPts[dPts.length - 1].x, dPts[dPts.length - 1].y);
                ctx.lineTo(dPts[0].x, dPts[0].y);
                ctx.stroke();
                ctx.restore();
            }

            if (dPts.length >= 2) {
                ctx.save();
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = pal.accent;
                ctx.lineWidth = 2.0;
                ctx.beginPath();
                for (var dl = 0; dl < dPts.length; dl += 1) {
                    if (dl === 0) ctx.moveTo(dPts[dl].x, dPts[dl].y);
                    else ctx.lineTo(dPts[dl].x, dPts[dl].y);
                }
                ctx.stroke();
                ctx.restore();
            }

            if (this._tool === "parking" && this._hoverPos && !this._dragState && this._hoverVertexIdx === -1) {
                var lastDraftPt = dPts[dPts.length - 1];
                ctx.save();
                ctx.setLineDash([3, 4]);
                ctx.strokeStyle = withAlpha(pal.accent, 0.45);
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.moveTo(lastDraftPt.x, lastDraftPt.y);
                ctx.lineTo(this._hoverPos.x, this._hoverPos.y);
                ctx.stroke();
                ctx.restore();
            }

            for (var dv = 0; dv < dPts.length; dv += 1) {
                var draftActive = this._tool === "parking" &&
                    ((this._dragState && this._dragState.target === "parking" && this._dragState.index === dv) ||
                     this._hoverVertexIdx === dv);
                ctx.fillStyle = pal.bgCard;
                ctx.strokeStyle = pal.accent;
                ctx.lineWidth = draftActive ? 2.4 : 1.8;
                ctx.beginPath();
                ctx.arc(dPts[dv].x, dPts[dv].y, draftActive ? 6.5 : 4.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }

            var draftLabelX = dPts.length >= 3 ? (dSumX / dPts.length) : dPts[0].x;
            var draftLabelY = dPts.length >= 3 ? (dSumY / dPts.length) : (dPts[0].y - 18);
            this._drawCenteredBadge(
                ctx,
                draftLabelX,
                draftLabelY,
                "禁停标定区",
                pal.bgCard,
                pal.accent,
                pal.accent,
                w,
                h
            );
        }

        // 3. Ground-correspondence calibration polygon & P1..PN vertices
        if (this._imagePoints.length > 0) {
            var cPts = [];
            for (var pi = 0; pi < this._imagePoints.length; pi += 1) {
                cPts.push(this.imageToCanvas(this._imagePoints[pi][0], this._imagePoints[pi][1]));
            }

            if (cPts.length >= 3) {
                ctx.save();
                ctx.beginPath();
                for (var pf = 0; pf < cPts.length; pf += 1) {
                    if (pf === 0) ctx.moveTo(cPts[pf].x, cPts[pf].y);
                    else ctx.lineTo(cPts[pf].x, cPts[pf].y);
                }
                ctx.closePath();
                ctx.fillStyle = withAlpha(pal.accent, 0.13);
                ctx.fill();
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = withAlpha(pal.accent, 0.90);
                ctx.lineWidth = 1.6;
                ctx.stroke();

                if (cPts.length === 4) {
                    ctx.setLineDash([3, 4]);
                    ctx.strokeStyle = withAlpha(pal.accent, 0.30);
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(cPts[0].x, cPts[0].y);
                    ctx.lineTo(cPts[2].x, cPts[2].y);
                    ctx.moveTo(cPts[1].x, cPts[1].y);
                    ctx.lineTo(cPts[3].x, cPts[3].y);
                    ctx.stroke();
                }
                ctx.restore();
            } else if (cPts.length === 2) {
                ctx.save();
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = withAlpha(pal.accent, 0.90);
                ctx.lineWidth = 1.6;
                ctx.beginPath();
                ctx.moveTo(cPts[0].x, cPts[0].y);
                ctx.lineTo(cPts[1].x, cPts[1].y);
                ctx.stroke();
                ctx.restore();
            }

            if (this._tool === "points" && this._hoverPos && !this._dragState && this._hoverVertexIdx === -1) {
                var lastPt = cPts[cPts.length - 1];
                ctx.save();
                ctx.setLineDash([3, 4]);
                ctx.strokeStyle = withAlpha(pal.accent, 0.42);
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.moveTo(lastPt.x, lastPt.y);
                ctx.lineTo(this._hoverPos.x, this._hoverPos.y);
                ctx.stroke();
                ctx.restore();
            }

            for (var vi = 0; vi < cPts.length; vi += 1) {
                var pt = cPts[vi];
                var isActive = this._tool === "points" &&
                    ((this._dragState && this._dragState.target === "points" && this._dragState.index === vi) ||
                     this._hoverVertexIdx === vi);

                if (isActive) {
                    ctx.fillStyle = withAlpha(pal.accent, 0.22);
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 11, 0, Math.PI * 2);
                    ctx.fill();
                }

                ctx.fillStyle = pal.bgCard;
                ctx.strokeStyle = pal.accent;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, isActive ? 7 : 5.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = pal.accent;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 2.2, 0, Math.PI * 2);
                ctx.fill();

                var badgeTxt = "P" + (vi + 1);
                ctx.save();
                ctx.font = "600 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
                var bPadX = 5;
                var bH = 17;
                var bW = ctx.measureText(badgeTxt).width + bPadX * 2;
                var bx = pt.x + 9;
                if (bx + bW > w - 6) bx = pt.x - bW - 9;
                bx = clamp(bx, 6, Math.max(6, w - bW - 6));
                var by = pt.y - bH - 6;
                if (by < 6) by = pt.y + 8;
                by = clamp(by, 6, Math.max(6, h - bH - 6));

                drawRoundedRectPath(ctx, bx, by, bW, bH, 4);
                ctx.fillStyle = withAlpha(pal.bgCard, 0.94);
                ctx.fill();
                ctx.lineWidth = 1;
                ctx.strokeStyle = isActive ? pal.accent : withAlpha(pal.accent, 0.72);
                ctx.stroke();

                ctx.fillStyle = pal.textPrimary;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillText(badgeTxt, bx + bPadX, by + bH * 0.5 + 0.5);
                ctx.restore();
            }
        }

        // Top-left compact HUD status pill when frame is loaded
        if (this._hasFrameDimensions() && w >= 220) {
            var hudText = this._tool === "parking"
                ? ("禁停区绘制 · 草稿 " + this._currentParkingDraft.length + " 点")
                : ("参考点标定 · 已选 " + this._imagePoints.length + " 点");
            hudText += " · " + this._imageWidth + "x" + this._imageHeight;
            drawPill(ctx, 10, 8, hudText, pal.bgCard, pal.hairline, pal.textSecondary, w);
        }

        ctx.restore();
    };

    CalibrationCanvas.prototype.loadImage = function (dataUrl, imgWidth, imgHeight) {
        if (this._destroyed) return;
        this._loadToken += 1;
        var token = this._loadToken;
        var reqW = toNum(imgWidth, 0);
        var reqH = toNum(imgHeight, 0);

        this._dragState = null;
        this._hoverVertexIdx = -1;

        if (!dataUrl || typeof dataUrl !== "string") {
            this._image = null;
            this._imageReady = false;
            this._imageWidth = reqW > 0 ? reqW : 0;
            this._imageHeight = reqH > 0 ? reqH : 0;
            this._measure();
            this._updateCursor();
            this._render();
            return;
        }

        this._image = null;
        this._imageReady = false;
        this._imageWidth = reqW > 0 ? reqW : 0;
        this._imageHeight = reqH > 0 ? reqH : 0;
        this._measure();
        this._updateCursor();
        this._render();

        if (typeof global.Image !== "function") {
            return;
        }

        var self = this;
        var img = new global.Image();
        img.onload = function () {
            if (self._destroyed || token !== self._loadToken) return;
            var natW = toNum(img.naturalWidth || img.width, 0);
            var natH = toNum(img.naturalHeight || img.height, 0);
            if (self._imageWidth <= 0 && natW > 0) self._imageWidth = natW;
            if (self._imageHeight <= 0 && natH > 0) self._imageHeight = natH;
            if (self._imageWidth <= 0) self._imageWidth = 1280;
            if (self._imageHeight <= 0) self._imageHeight = 720;
            self._image = img;
            self._imageReady = true;
            self._measure();
            self._updateCursor();
            self._render();
        };
        img.onerror = function () {
            if (self._destroyed || token !== self._loadToken) return;
            self._image = null;
            self._imageReady = false;
            self._render();
        };
        img.src = dataUrl;
    };

    CalibrationCanvas.prototype.setTool = function (tool) {
        if (this._destroyed) return;
        this._tool = tool === "parking" ? "parking" : "points";
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._updateCursor();
        this._render();
    };

    CalibrationCanvas.prototype.undo = function () {
        if (this._destroyed) return false;
        var list = this._activeVertexList();
        if (list.length === 0) return false;
        list.pop();
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._updateCursor();
        this._render();
        this._emitChange(true);
        return true;
    };

    CalibrationCanvas.prototype.finishParkingZone = function () {
        if (this._destroyed) return false;
        if (this._currentParkingDraft.length < 3) {
            return false;
        }
        this._parkingZonesImage.push(clonePointList(this._currentParkingDraft));
        this._currentParkingDraft = [];
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._updateCursor();
        this._render();
        this._emitChange(true);
        return true;
    };

    CalibrationCanvas.prototype.clearPoints = function () {
        if (this._destroyed) return;
        this._imagePoints = [];
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._updateCursor();
        this._render();
        this._emitChange(true);
    };

    CalibrationCanvas.prototype.clearParkingZones = function () {
        if (this._destroyed) return;
        this._parkingZonesImage = [];
        this._currentParkingDraft = [];
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._updateCursor();
        this._render();
        this._emitChange(true);
    };

    CalibrationCanvas.prototype.getState = function () {
        return {
            imagePoints: clonePointList(this._imagePoints),
            parkingZonesImage: clonePolygonList(this._parkingZonesImage),
            currentParkingDraft: clonePointList(this._currentParkingDraft),
            imageWidth: this._imageWidth,
            imageHeight: this._imageHeight
        };
    };

    CalibrationCanvas.prototype.setState = function (nextState) {
        if (this._destroyed || !nextState || typeof nextState !== "object") return;
        if (nextState.imagePoints !== undefined) {
            this._imagePoints = clonePointList(nextState.imagePoints);
        }
        if (nextState.parkingZonesImage !== undefined) {
            this._parkingZonesImage = clonePolygonList(nextState.parkingZonesImage);
        }
        if (nextState.currentParkingDraft !== undefined) {
            this._currentParkingDraft = clonePointList(nextState.currentParkingDraft);
        }
        if (isFiniteNum(nextState.imageWidth) && nextState.imageWidth > 0) {
            this._imageWidth = nextState.imageWidth;
        }
        if (isFiniteNum(nextState.imageHeight) && nextState.imageHeight > 0) {
            this._imageHeight = nextState.imageHeight;
        }
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._measure();
        this._updateCursor();
        this._render();
    };

    CalibrationCanvas.prototype.refreshTheme = function () {
        if (this._destroyed) return;
        this._palette = readThemePalette(this._probe);
        this._measure();
        this._render();
    };

    CalibrationCanvas.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        this._loadToken += 1;
        this._image = null;
        if (this._emitTimer !== null && typeof global.clearTimeout === "function") {
            global.clearTimeout(this._emitTimer);
            this._emitTimer = null;
        }
        this._canvas.removeEventListener("pointerdown", this._onPointerDown);
        this._canvas.removeEventListener("pointermove", this._onPointerMove);
        this._canvas.removeEventListener("pointerup", this._onPointerUp);
        this._canvas.removeEventListener("pointercancel", this._onPointerCancel);
        this._canvas.removeEventListener("pointerleave", this._onPointerLeave);
        detachResizeObserver(this);
        this._unobserveTheme();
        this._motionTracker.destroy();
        if (this._probe && this._probe.parentNode) {
            this._probe.parentNode.removeChild(this._probe);
        }
    };

    /* ======================================================================
     * 6. AlertAudioSynth (Zero-Asset Web Audio L3 Instrumentation Chime)
     * ====================================================================== */
    function AlertAudioSynth() {
        if (!(this instanceof AlertAudioSynth)) {
            return new AlertAudioSynth();
        }
        this._enabled = false;
        this._ctx = null;
        this._lastChimeMs = -Infinity;
        this._lastRiskLevel = 0;
        this._destroyed = false;
    }

    AlertAudioSynth.prototype._ensureAudioContext = function () {
        if (this._destroyed) return null;
        if (!this._ctx) {
            var AudioCtx = global.AudioContext || global.webkitAudioContext;
            if (typeof AudioCtx !== "function") {
                return null;
            }
            try {
                this._ctx = new AudioCtx();
            } catch (_err) {
                this._ctx = null;
                return null;
            }
        }
        if (this._ctx && this._ctx.state === "suspended" && typeof this._ctx.resume === "function") {
            try {
                var p = this._ctx.resume();
                if (p && typeof p.catch === "function") {
                    p.catch(function () {});
                }
            } catch (_err2) {
                // Ignore resume failure
            }
        }
        return this._ctx;
    };

    AlertAudioSynth.prototype.setEnabled = function (enabled) {
        if (this._destroyed) return false;
        var next = Boolean(enabled);
        this._enabled = next;
        if (next) {
            this._ensureAudioContext();
        } else {
            this._lastRiskLevel = 0;
        }
        return this._enabled;
    };

    AlertAudioSynth.prototype.isEnabled = function () {
        return this._enabled;
    };

    AlertAudioSynth.prototype._playDualToneChime = function (ctx) {
        if (!ctx || ctx.state === "closed" ||
            typeof ctx.createOscillator !== "function" ||
            typeof ctx.createGain !== "function") {
            return false;
        }
        try {
            var t0 = typeof ctx.currentTime === "number" ? ctx.currentTime : 0;
            var notes = [
                { freq: 784.0, delay: 0.0, dur: 0.20, peak: 0.05 },
                { freq: 1046.5, delay: 0.095, dur: 0.26, peak: 0.055 }
            ];
            for (var i = 0; i < notes.length; i += 1) {
                var spec = notes[i];
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                var startAt = t0 + spec.delay;
                var stopAt = startAt + spec.dur;

                osc.type = "sine";
                osc.frequency.setValueAtTime(spec.freq, startAt);

                gain.gain.setValueAtTime(0.0001, startAt);
                gain.gain.linearRampToValueAtTime(spec.peak, startAt + 0.014);
                gain.gain.exponentialRampToValueAtTime(0.0001, stopAt);

                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start(startAt);
                osc.stop(stopAt + 0.01);
            }
            return true;
        } catch (_err) {
            return false;
        }
    };

    AlertAudioSynth.prototype.triggerRisk = function (riskLevel) {
        if (this._destroyed) return false;
        if (!this._enabled) {
            this._lastRiskLevel = 0;
            return false;
        }

        var lvl = parseRiskLevel(riskLevel);
        var prevLvl = this._lastRiskLevel;
        this._lastRiskLevel = lvl;

        if (lvl < 3 || prevLvl >= 3) {
            return false;
        }

        var nowMs = Date.now();
        if (nowMs < this._lastChimeMs) {
            this._lastChimeMs = nowMs - 2500;
        }
        if (nowMs - this._lastChimeMs < 2500) {
            return false;
        }

        var ctx = this._ensureAudioContext();
        if (!ctx) return false;

        if (this._playDualToneChime(ctx)) {
            this._lastChimeMs = nowMs;
            return true;
        }
        return false;
    };

    AlertAudioSynth.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        this._enabled = false;
        if (this._ctx && typeof this._ctx.close === "function" && this._ctx.state !== "closed") {
            try {
                var p = this._ctx.close();
                if (p && typeof p.catch === "function") {
                    p.catch(function () {});
                }
            } catch (_err) {
                // Ignore close failure
            }
        }
        this._ctx = null;
    };

    /* ======================================================================
     * 7. SimulatedCameraCanvas (Synchronized 3D Oblique Perspective Feed)
     * ====================================================================== */
    function SimulatedCameraCanvas(canvasEl, options) {
        if (!(this instanceof SimulatedCameraCanvas)) {
            return new SimulatedCameraCanvas(canvasEl, options);
        }
        if (!canvasEl) {
            throw new Error("SimulatedCameraCanvas requires a valid <canvas> element");
        }
        var opts = options || {};
        this._canvas = canvasEl;
        this._ctx = canvasEl.getContext("2d");
        this._destroyed = false;
        this._cssWidth = 0;
        this._cssHeight = 0;
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        this._lastState = opts.initialState || null;

        this._probe = createColorProbe();
        this._palette = readThemePalette(this._probe);

        var self = this;
        this._unobserveTheme = observeThemeChanges(function () {
            self.refreshTheme();
        });
        attachResizeObserver(this._canvas, this, function () {
            self._measure();
            self._render();
        });
        this._measure();
        this._render();
    }

    SimulatedCameraCanvas.prototype._measure = function () {
        var rect = this._canvas.getBoundingClientRect();
        var cssW = Math.max(0, Math.round(rect.width || this._canvas.clientWidth || 0));
        var cssH = Math.max(0, Math.round(rect.height || this._canvas.clientHeight || 0));
        if (cssW < 2 || cssH < 2) {
            cssW = 540;
            cssH = 340;
        }
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        var pxW = Math.round(cssW * this._dpr);
        var pxH = Math.round(cssH * this._dpr);
        if (this._canvas.width !== pxW || this._canvas.height !== pxH) {
            this._canvas.width = pxW;
            this._canvas.height = pxH;
        }
        this._cssWidth = cssW;
        this._cssHeight = cssH;
    };

    SimulatedCameraCanvas.prototype.renderState = function (state) {
        if (this._destroyed) return;
        this._lastState = state || null;
        this._render();
    };

    SimulatedCameraCanvas.prototype.refreshTheme = function () {
        if (this._destroyed) return;
        this._palette = readThemePalette(this._probe);
        this._measure();
        this._render();
    };

    SimulatedCameraCanvas.prototype._project3D = function (wx, wy, wz, cam) {
        var dx = wx - cam.x;
        var dy = wy - cam.y;
        var dz = wz - cam.z;
        var zCam = dy * cam.fwdY + dz * cam.fwdZ;
        if (zCam <= 0.45) return null;
        var xCam = dx;
        var yCam = dy * cam.upY + dz * cam.upZ;
        return {
            x: cam.cx + (xCam / zCam) * cam.focal,
            y: cam.cy - (yCam / zCam) * cam.focal,
            zCam: zCam
        };
    };

    function drawCornerBracketBox(ctx, x1, y1, bw, bh, color, lineWidth) {
        var arm = clamp(Math.min(bw, bh) * 0.24, 5, 14);
        var x2 = x1 + bw;
        var y2 = y1 + bh;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth || 1.6;
        ctx.lineCap = "square";
        ctx.beginPath();
        ctx.moveTo(x1, y1 + arm);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1 + arm, y1);
        ctx.moveTo(x2 - arm, y1);
        ctx.lineTo(x2, y1);
        ctx.lineTo(x2, y1 + arm);
        ctx.moveTo(x2, y2 - arm);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x2 - arm, y2);
        ctx.moveTo(x1 + arm, y2);
        ctx.lineTo(x1, y2);
        ctx.lineTo(x1, y2 - arm);
        ctx.stroke();
        ctx.restore();
    }

    function drawProjectedQuad(ctx, pts, fillStyle, strokeStyle, lineWidth) {
        if (!pts || pts.length < 3) return;
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (var i = 1; i < pts.length; i += 1) {
            ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.closePath();
        if (fillStyle) {
            ctx.fillStyle = fillStyle;
            ctx.fill();
        }
        if (strokeStyle) {
            ctx.strokeStyle = strokeStyle;
            ctx.lineWidth = lineWidth || 1;
            ctx.stroke();
        }
    }

    SimulatedCameraCanvas.prototype._render = function () {
        if (!this._ctx || this._cssWidth < 2 || this._cssHeight < 2) return;
        var ctx = this._ctx;
        var pal = this._palette;
        var w = this._cssWidth;
        var h = this._cssHeight;
        var st = this._lastState || {
            bounds: { xMin: -3.2, xMax: 3.2, yMin: 0.0, yMax: 15.0 },
            parkingZones: [],
            vehicles: [],
            vrus: [],
            conflicts: [],
            maxRisk: 0,
            speedModel: "ctrv",
            label: "SIM"
        };
        var b = st.bounds || { xMin: -3.2, xMax: 3.2, yMin: 0.0, yMax: 15.0 };

        // Calibrated oblique roadside CCTV camera looking along +Y across crosswalk
        var tgtY = 0.52 * (b.yMin + b.yMax);
        var camY = b.yMin - 8.5;
        var camZ = 6.2;
        var dY = tgtY - camY;
        var dZ = -camZ;
        var len = Math.hypot(dY, dZ) || 1;
        var fwdY = dY / len;
        var fwdZ = dZ / len;
        var cam = {
            x: 0.5 * (b.xMin + b.xMax),
            y: camY,
            z: camZ,
            fwdY: fwdY,
            fwdZ: fwdZ,
            upY: -fwdZ,
            upZ: fwdY,
            cx: w * 0.5,
            cy: h * 0.46,
            focal: Math.min(w * 0.92, h * 1.42)
        };

        var self = this;
        function p3(wx, wy, wz) {
            return self._project3D(wx, wy, wz || 0, cam);
        }

        function fillPolyPts(worldPts, fillColor, strokeColor, lineWidth, dashed) {
            var pts = [];
            for (var i = 0; i < worldPts.length; i += 1) {
                var pt = p3(worldPts[i][0], worldPts[i][1], worldPts[i][2] || 0);
                if (!pt) return;
                pts.push(pt);
            }
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(pts[0].x, pts[0].y);
            for (var j = 1; j < pts.length; j += 1) {
                ctx.lineTo(pts[j].x, pts[j].y);
            }
            ctx.closePath();
            if (fillColor) {
                ctx.fillStyle = fillColor;
                ctx.fill();
            }
            if (strokeColor) {
                if (dashed) ctx.setLineDash([5, 4]);
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = lineWidth || 1;
                ctx.stroke();
            }
            ctx.restore();
        }

        function drawLowPolyVehicle(v, vColor) {
            var hl = (v.lengthM || 4.5) * 0.5;
            var hw = (v.widthM || 1.9) * 0.5;
            var vh = v.kind === "bus" ? 2.85 : (v.kind === "truck" ? 2.55 : (v.kind === "suv" ? 1.66 : 1.46));
            var beltZ = v.kind === "bus" ? 1.32 : vh * 0.56;
            var hRad = resolveHeadingRad(v.headingRad, Math.PI * 0.5);
            var cosH = Math.cos(hRad);
            var sinH = Math.sin(hRad);

            function vPt(lx, ly, lz) {
                return p3(v.x + lx * cosH - ly * sinH, v.y + lx * sinH + ly * cosH, lz);
            }

            // Ground contact shadow
            fillPolyPts(
                [
                    [v.x + hl * 1.06 * cosH - hw * 1.08 * sinH, v.y + hl * 1.06 * sinH + hw * 1.08 * cosH, 0.01],
                    [v.x + hl * 1.06 * cosH + hw * 1.08 * sinH, v.y + hl * 1.06 * sinH - hw * 1.08 * cosH, 0.01],
                    [v.x - hl * 1.06 * cosH + hw * 1.08 * sinH, v.y - hl * 1.06 * sinH - hw * 1.08 * cosH, 0.01],
                    [v.x - hl * 1.06 * cosH - hw * 1.08 * sinH, v.y - hl * 1.06 * sinH + hw * 1.08 * cosH, 0.01]
                ],
                pal.isLight ? "rgba(18, 22, 30, 0.22)" : "rgba(0, 0, 0, 0.44)",
                null,
                0,
                false
            );

            var cFL0 = vPt(hl, hw, 0.14);
            var cFR0 = vPt(hl, -hw, 0.14);
            var cRR0 = vPt(-hl, -hw, 0.14);
            var cRL0 = vPt(-hl, hw, 0.14);
            var cFL1 = vPt(hl, hw, beltZ);
            var cFR1 = vPt(hl, -hw, beltZ);
            var cRR1 = vPt(-hl, -hw, beltZ);
            var cRL1 = vPt(-hl, hw, beltZ);

            var isTall = v.kind === "bus" || v.kind === "truck";
            var cabF = isTall ? (v.kind === "bus" ? hl * 0.90 : hl * 0.88) : hl * 0.36;
            var cabB = isTall ? (v.kind === "bus" ? -hl * 0.88 : hl * 0.10) : -hl * 0.72;
            var roofF = isTall ? cabF * 0.96 : hl * 0.16;
            var roofB = isTall ? cabB : -hl * 0.54;
            var roofHw = isTall ? hw * 0.96 : hw * 0.82;

            var gFL0 = vPt(cabF, hw * 0.94, beltZ);
            var gFR0 = vPt(cabF, -hw * 0.94, beltZ);
            var gRR0 = vPt(cabB, -hw * 0.94, beltZ);
            var gRL0 = vPt(cabB, hw * 0.94, beltZ);
            var gFL1 = vPt(roofF, roofHw, vh);
            var gFR1 = vPt(roofF, -roofHw, vh);
            var gRR1 = vPt(roofB, -roofHw, vh);
            var gRL1 = vPt(roofB, roofHw, vh);

            var allCheck = [cFL0, cFR0, cRR0, cRL0, cFL1, cFR1, cRR1, cRL1, gFL0, gFR0, gRR0, gRL0, gFL1, gFR1, gRR1, gRL1];
            for (var ci = 0; ci < allCheck.length; ci += 1) {
                if (!allCheck[ci]) return null;
            }

            var bodyBase = v.inParkingZone
                ? (pal.isLight ? "#8a92a0" : "#3a3f4a")
                : (pal.isLight ? "#323c4c" : "#444e5e");
            var bodySide = v.inParkingZone
                ? (pal.isLight ? "#737b88" : "#2f343e")
                : (pal.isLight ? "#28303e" : "#38414f");
            var bodyDeck = v.inParkingZone
                ? (pal.isLight ? "#9aa2af" : "#474d59")
                : (pal.isLight ? "#445063" : "#566275");
            var glassFill = pal.isLight ? "rgba(22, 34, 52, 0.74)" : "rgba(14, 22, 34, 0.88)";
            var edgeCol = withAlpha(vColor, v.inParkingZone ? 0.55 : 0.85);

            var faces = [];
            function pushFace(pts, fill, stroke, tag) {
                var zSum = 0;
                for (var fi = 0; fi < pts.length; fi += 1) zSum += pts[fi].zCam;
                faces.push({ pts: pts, fill: fill, stroke: stroke, zAvg: zSum / pts.length, tag: tag || "" });
            }

            // Lower chassis skirt faces
            pushFace([cFL0, cFR0, cFR1, cFL1], bodyBase, edgeCol, "front");
            pushFace([cRL0, cRR0, cRR1, cRL1], bodyBase, edgeCol, "rear");
            pushFace([cFL0, cRL0, cRL1, cFL1], bodySide, edgeCol, "left");
            pushFace([cFR0, cRR0, cRR1, cFR1], bodySide, edgeCol, "right");
            pushFace([cFL1, cFR1, cRR1, cRL1], bodyDeck, edgeCol, "deck");

            // Upper greenhouse / tilted windshield & roof faces
            pushFace([gFL0, gFR0, gFR1, gFL1], glassFill, edgeCol, "windshield");
            pushFace([gRL0, gRR0, gRR1, gRL1], glassFill, edgeCol, "rearGlass");
            pushFace([gFL0, gRL0, gRL1, gFL1], glassFill, edgeCol, "leftGlass");
            pushFace([gFR0, gRR0, gRR1, gFR1], glassFill, edgeCol, "rightGlass");
            pushFace([gFL1, gFR1, gRR1, gRL1], bodyDeck, edgeCol, "roof");

            faces.sort(function (fa, fb) { return fb.zAvg - fa.zAvg; });

            ctx.save();
            for (var fk = 0; fk < faces.length; fk += 1) {
                drawProjectedQuad(ctx, faces[fk].pts, faces[fk].fill, faces[fk].stroke, 1.0);
            }

            // Warm front headlights
            var hlL = vPt(hl + 0.01, hw * 0.66, beltZ * 0.56);
            var hlR = vPt(hl + 0.01, -hw * 0.66, beltZ * 0.56);
            var frontFacePts = [cFL0, cFR0, cFR1, cFL1];
            var frontZ = (frontFacePts[0].zCam + frontFacePts[1].zCam) * 0.5;
            var rearZ = (cRL0.zCam + cRR0.zCam) * 0.5;
            if (hlL && hlR && frontZ <= rearZ + 0.8) {
                var headR = clamp((0.14 / hlL.zCam) * cam.focal, 2.0, 5.0);
                ctx.fillStyle = "#fde68a";
                ctx.beginPath();
                ctx.arc(hlL.x, hlL.y, headR, 0, Math.PI * 2);
                ctx.arc(hlR.x, hlR.y, headR, 0, Math.PI * 2);
                ctx.fill();
            }

            // Semantic brake-light bar & tail lamps when decelerating (accelMs2 < -1.0)
            var isBraking = !v.inParkingZone && isFiniteNum(v.accelMs2) && v.accelMs2 < -1.0;
            if (rearZ <= frontZ + 1.2) {
                var blL0 = vPt(-hl - 0.01, hw * 0.80, beltZ * 0.60);
                var blR0 = vPt(-hl - 0.01, -hw * 0.80, beltZ * 0.60);
                var blL1 = vPt(-hl - 0.01, hw * 0.80, beltZ * 0.78);
                var blR1 = vPt(-hl - 0.01, -hw * 0.80, beltZ * 0.78);
                if (isBraking && blL0 && blR0 && blL1 && blR1) {
                    drawProjectedQuad(ctx, [blL0, blR0, blR1, blL1], pal.critical, "#ff8b8f", 0.9);
                } else {
                    var tlL = vPt(-hl - 0.01, hw * 0.66, beltZ * 0.68);
                    var tlR = vPt(-hl - 0.01, -hw * 0.66, beltZ * 0.68);
                    if (tlL && tlR) {
                        var tailR = clamp((0.11 / tlL.zCam) * cam.focal, 1.8, 4.0);
                        ctx.fillStyle = withAlpha(pal.critical, 0.55);
                        ctx.beginPath();
                        ctx.arc(tlL.x, tlL.y, tailR, 0, Math.PI * 2);
                        ctx.arc(tlR.x, tlR.y, tailR, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            }
            ctx.restore();

            var minBx = Infinity;
            var minBy = Infinity;
            var maxBx = -Infinity;
            var maxBy = -Infinity;
            for (var bi = 0; bi < allCheck.length; bi += 1) {
                if (allCheck[bi].x < minBx) minBx = allCheck[bi].x;
                if (allCheck[bi].y < minBy) minBy = allCheck[bi].y;
                if (allCheck[bi].x > maxBx) maxBx = allCheck[bi].x;
                if (allCheck[bi].y > maxBy) maxBy = allCheck[bi].y;
            }
            return { x: minBx, y: minBy, w: Math.max(12, maxBx - minBx), h: Math.max(12, maxBy - minBy) };
        }

        function drawGroundedVruSilhouette(u, uColor, isBike) {
            var fillBody = pal.isLight ? "#2a3342" : "#cfd6e2";
            var fillBikeFrame = pal.isLight ? "#3a4558" : "#9aa6ba";
            if (isBike) {
                var psi = u.headingRad !== undefined ? u.headingRad : Math.atan2(u.vy || 0, u.vx || 1);
                var cosB = Math.cos(psi);
                var sinB = Math.sin(psi);
                function bPt(lx, ly, lz) {
                    return p3(u.x + lx * cosB - ly * sinB, u.y + lx * sinB + ly * cosB, lz);
                }
                var rw = bPt(-0.62, 0, 0.32);
                var fw = bPt(0.62, 0, 0.32);
                var seat = bPt(-0.18, 0, 0.90);
                var bar = bPt(0.38, 0, 1.00);
                var sh = bPt(0.05, 0, 1.32);
                var head = bPt(0.12, 0, 1.54);
                var gBack = bPt(-0.82, 0, 0);
                var gFront = bPt(0.82, 0, 0);
                if (!rw || !fw || !seat || !bar || !sh || !head || !gBack || !gFront) return null;

                // Ground shadow
                fillPolyPts(
                    [
                        [u.x - 0.8 * cosB - 0.28 * sinB, u.y - 0.8 * sinB + 0.28 * cosB, 0.01],
                        [u.x + 0.8 * cosB - 0.28 * sinB, u.y + 0.8 * sinB + 0.28 * cosB, 0.01],
                        [u.x + 0.8 * cosB + 0.28 * sinB, u.y + 0.8 * sinB - 0.28 * cosB, 0.01],
                        [u.x - 0.8 * cosB + 0.28 * sinB, u.y - 0.8 * sinB - 0.28 * cosB, 0.01]
                    ],
                    pal.isLight ? "rgba(18, 22, 30, 0.22)" : "rgba(0, 0, 0, 0.42)",
                    null,
                    0,
                    false
                );

                var wheelR = clamp((0.32 / rw.zCam) * cam.focal, 4, 15);
                var headR = clamp((0.13 / head.zCam) * cam.focal, 2.5, 8);
                ctx.save();
                ctx.strokeStyle = fillBikeFrame;
                ctx.lineWidth = 1.8;
                ctx.beginPath();
                ctx.arc(rw.x, rw.y, wheelR, 0, Math.PI * 2);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(fw.x, fw.y, wheelR, 0, Math.PI * 2);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(rw.x, rw.y);
                ctx.lineTo(seat.x, seat.y);
                ctx.lineTo(bar.x, bar.y);
                ctx.lineTo(fw.x, fw.y);
                ctx.stroke();

                // Rider silhouette
                ctx.strokeStyle = fillBody;
                ctx.lineWidth = 2.6;
                ctx.lineCap = "round";
                ctx.beginPath();
                ctx.moveTo(seat.x, seat.y);
                ctx.lineTo(sh.x, sh.y);
                ctx.lineTo(bar.x, bar.y);
                ctx.stroke();

                ctx.fillStyle = fillBody;
                ctx.beginPath();
                ctx.arc(head.x, head.y, headR, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();

                var xs = [rw.x - wheelR, fw.x + wheelR, gBack.x, gFront.x, head.x];
                var ys = [head.y - headR, rw.y + wheelR, fw.y + wheelR, gBack.y, gFront.y];
                var x1 = Math.min.apply(null, xs) - 3;
                var x2 = Math.max.apply(null, xs) + 3;
                var y1 = Math.min.apply(null, ys) - 3;
                var y2 = Math.max.apply(null, ys) + 2;
                return { x: x1, y: y1, w: Math.max(16, x2 - x1), h: Math.max(24, y2 - y1) };
            }

            // Pedestrian grounded silhouette
            var pFootL = p3(u.x - 0.14, u.y, 0);
            var pFootR = p3(u.x + 0.14, u.y, 0);
            var pHipL = p3(u.x - 0.13, u.y, 0.80);
            var pHipR = p3(u.x + 0.13, u.y, 0.80);
            var pShL = p3(u.x - 0.21, u.y, 1.36);
            var pShR = p3(u.x + 0.21, u.y, 1.36);
            var pHead = p3(u.x, u.y, 1.58);
            if (!pFootL || !pFootR || !pHipL || !pHipR || !pShL || !pShR || !pHead) return null;

            fillPolyPts(
                [
                    [u.x - 0.26, u.y - 0.26, 0.01],
                    [u.x + 0.26, u.y - 0.26, 0.01],
                    [u.x + 0.26, u.y + 0.26, 0.01],
                    [u.x - 0.26, u.y + 0.26, 0.01]
                ],
                pal.isLight ? "rgba(18, 22, 30, 0.22)" : "rgba(0, 0, 0, 0.42)",
                null,
                0,
                false
            );

            var pHeadR = clamp((0.135 / pHead.zCam) * cam.focal, 2.5, 8.5);
            ctx.save();
            // Legs
            ctx.strokeStyle = fillBody;
            ctx.lineWidth = clamp((0.10 / pFootL.zCam) * cam.focal, 1.8, 4.0);
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(pFootL.x, pFootL.y);
            ctx.lineTo(pHipL.x, pHipL.y);
            ctx.moveTo(pFootR.x, pFootR.y);
            ctx.lineTo(pHipR.x, pHipR.y);
            ctx.stroke();

            // Torso
            drawProjectedQuad(ctx, [pHipL, pHipR, pShR, pShL], fillBody, withAlpha(uColor, 0.75), 0.9);

            // Head
            ctx.fillStyle = fillBody;
            ctx.beginPath();
            ctx.arc(pHead.x, pHead.y, pHeadR, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();

            var px1 = Math.min(pShL.x, pFootL.x, pHead.x - pHeadR) - 4;
            var px2 = Math.max(pShR.x, pFootR.x, pHead.x + pHeadR) + 4;
            var py1 = pHead.y - pHeadR - 3;
            var py2 = Math.max(pFootL.y, pFootR.y) + 2;
            return { x: px1, y: py1, w: Math.max(14, px2 - px1), h: Math.max(24, py2 - py1) };
        }

        ctx.save();
        ctx.scale(this._dpr, this._dpr);
        ctx.clearRect(0, 0, w, h);

        // Clip all 3D scene rendering strictly inside (0, 0, w, h)
        ctx.beginPath();
        ctx.rect(0, 0, w, h);
        ctx.clip();

        // Background sky / campus horizon band + distant ground apron
        ctx.fillStyle = pal.isLight ? "#dde1e8" : "#101012";
        ctx.fillRect(0, 0, w, h);

        var horizonPt = p3(cam.x, b.yMax + 80, 0);
        var horizonY = horizonPt ? clamp(horizonPt.y, 18, h * 0.32) : h * 0.22;
        ctx.fillStyle = pal.roadShoulder;
        ctx.fillRect(0, horizonY, w, h - horizonY);

        // Crisp campus horizon curb + distant light pole silhouettes
        ctx.strokeStyle = pal.hairline;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, horizonY);
        ctx.lineTo(w, horizonY);
        ctx.stroke();

        var poleXs = [b.xMin - 0.6, b.xMax + 0.6];
        var poleYs = [5.5, 11.8];
        for (var pxi = 0; pxi < poleXs.length; pxi += 1) {
            for (var pyi = 0; pyi < poleYs.length; pyi += 1) {
                var pb = p3(poleXs[pxi], poleYs[pyi], 0);
                var ptTop = p3(poleXs[pxi], poleYs[pyi], 4.0);
                var ptArm = p3(poleXs[pxi] * 0.78, poleYs[pyi], 3.9);
                if (pb && ptTop && ptArm) {
                    ctx.strokeStyle = withAlpha(pal.textMuted, 0.34);
                    ctx.lineWidth = 1.6;
                    ctx.beginPath();
                    ctx.moveTo(pb.x, pb.y);
                    ctx.lineTo(ptTop.x, ptTop.y);
                    ctx.lineTo(ptArm.x, ptArm.y);
                    ctx.stroke();
                }
            }
        }

        // Far-extending asphalt carriageway & sidewalks (farY aligned with scenario exit boundary)
        var farY = b.yMax + 2.6;
        var nearY = Math.max(camY + 1.2, b.yMin - 1.5);
        fillPolyPts(
            [[b.xMin - 1.8, nearY], [b.xMax + 1.8, nearY], [b.xMax + 1.8, farY], [b.xMin - 1.8, farY]],
            withAlpha(pal.bgRaised, pal.isLight ? 0.85 : 0.7),
            null,
            0,
            false
        );
        fillPolyPts(
            [[b.xMin, nearY], [b.xMax, nearY], [b.xMax, farY], [b.xMin, farY]],
            pal.roadSurface,
            pal.roadCurb,
            1.4,
            false
        );

        // Perspective dashed center divider
        var cxMid = 0.5 * (b.xMin + b.xMax);
        ctx.save();
        ctx.strokeStyle = pal.roadMark;
        ctx.lineWidth = 1.4;
        for (var ly = nearY + 0.4; ly < farY - 0.8; ly += 1.8) {
            var d0 = p3(cxMid, ly, 0);
            var d1 = p3(cxMid, Math.min(farY, ly + 0.9), 0);
            if (d0 && d1) {
                ctx.beginPath();
                ctx.moveTo(d0.x, d0.y);
                ctx.lineTo(d1.x, d1.y);
                ctx.stroke();
            }
        }
        ctx.restore();

        // Crosswalk perspective zebra stripes (y = 8.80 .. 10.00, matching twin & scenarios)
        var cwY0 = 8.80;
        var cwY1 = 10.00;
        if (cwY1 >= b.yMin + 0.4 && cwY0 <= b.yMax - 0.4) {
            var cwTint = withAlpha(pal.textPrimary, pal.isLight ? 0.16 : 0.12);
            var cwClampedY0 = Math.max(b.yMin, cwY0);
            var cwClampedY1 = Math.min(b.yMax, cwY1);
            for (var zx = b.xMin + 0.4; zx < b.xMax - 0.35; zx += 0.72) {
                fillPolyPts(
                    [[zx, cwClampedY0], [Math.min(zx + 0.38, b.xMax - 0.3), cwClampedY0], [Math.min(zx + 0.38, b.xMax - 0.3), cwClampedY1], [zx, cwClampedY1]],
                    cwTint,
                    null,
                    0,
                    false
                );
            }
        }

        // Calibrated 4-point homography ROI trapezoid overlay
        var roiPadX = 0.32;
        var roiY0 = b.yMin + 0.8;
        var roiY1 = b.yMax - 0.8;
        var roiWorld = [
            [b.xMin + roiPadX, roiY1, 0],
            [b.xMax - roiPadX, roiY1, 0],
            [b.xMax - roiPadX, roiY0, 0],
            [b.xMin + roiPadX, roiY0, 0]
        ];
        fillPolyPts(roiWorld, withAlpha(pal.accent, 0.045), withAlpha(pal.accent, 0.46), 1.2, false);
        for (var rci = 0; rci < roiWorld.length; rci += 1) {
            var rpt = p3(roiWorld[rci][0], roiWorld[rci][1], 0);
            if (rpt) {
                ctx.fillStyle = pal.accent;
                ctx.beginPath();
                ctx.arc(rpt.x, rpt.y, 2.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Roadside parking exclusion zone projected onto pavement
        var zones = st.parkingZones || [];
        for (var zi = 0; zi < zones.length; zi += 1) {
            var poly = zones[zi];
            if (!Array.isArray(poly) || poly.length < 3) continue;
            var worldPoly = poly.map(function (pt) { return [pt[0], pt[1], 0]; });
            fillPolyPts(worldPoly, withAlpha(pal.parkingZone, 0.10), withAlpha(pal.parkingZone, 0.72), 1.3, true);
        }

        // Future kinematic trajectory arcs projected onto pavement (honoring speedModel)
        var vehicles = Array.isArray(st.vehicles) ? st.vehicles : [];
        for (var vi = 0; vi < vehicles.length; vi += 1) {
            var veh = vehicles[vi];
            var spdMs = Math.hypot(veh.vx || 0, veh.vy || 0);
            if (spdMs < 0.35 || veh.inParkingZone) continue;
            var arcPts = predictCtrvArc(veh, 2.5, 16, st.speedModel);
            var arcColor = veh.riskLevel >= 1 ? riskColor(pal, veh.riskLevel) : pal.safe;
            ctx.save();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = withAlpha(arcColor, 0.86);
            ctx.lineWidth = 1.7;
            ctx.beginPath();
            var startedArc = false;
            for (var ai = 0; ai < arcPts.length; ai += 1) {
                var aProj = p3(arcPts[ai].x, arcPts[ai].y, 0.02);
                if (!aProj) continue;
                if (!startedArc) {
                    ctx.moveTo(aProj.x, aProj.y);
                    startedArc = true;
                } else {
                    ctx.lineTo(aProj.x, aProj.y);
                }
            }
            if (startedArc) ctx.stroke();
            ctx.restore();
        }

        // Projected TTC convergence link & ground ring
        var vrus = Array.isArray(st.vrus) ? st.vrus : [];
        var conflicts = Array.isArray(st.conflicts) ? st.conflicts : [];
        for (var ci = 0; ci < conflicts.length; ci += 1) {
            var cf = conflicts[ci];
            if ((cf.riskLevel || 0) < 2) continue;
            var cfColor = riskColor(pal, cf.riskLevel);
            var vObj = findById(vehicles, cf.vehTrackId);
            var uObj = findById(vrus, cf.pedTrackId);
            if (vObj && uObj) {
                var vp = p3(vObj.x, vObj.y, 0.05);
                var up = p3(uObj.x, uObj.y, 0.05);
                if (vp && up) {
                    ctx.save();
                    ctx.setLineDash([4, 3]);
                    ctx.strokeStyle = withAlpha(cfColor, 0.85);
                    ctx.lineWidth = 1.4;
                    ctx.beginPath();
                    ctx.moveTo(vp.x, vp.y);
                    ctx.lineTo(up.x, up.y);
                    ctx.stroke();
                    ctx.restore();
                }
            }
            var cxy = cf.conflictGroundXY;
            if ((!Array.isArray(cxy) || cxy.length < 2) && vObj && uObj) {
                cxy = [(vObj.x + uObj.x) * 0.5, (vObj.y + uObj.y) * 0.5];
            }
            if (Array.isArray(cxy) && cxy.length >= 2) {
                var ringPts = [];
                for (var ang = 0; ang < 16; ang += 1) {
                    var theta = (ang / 16) * Math.PI * 2;
                    var rp = p3(cxy[0] + Math.cos(theta) * 0.65, cxy[1] + Math.sin(theta) * 0.65, 0.03);
                    if (rp) ringPts.push(rp);
                }
                if (ringPts.length >= 6) {
                    ctx.strokeStyle = cfColor;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.moveTo(ringPts[0].x, ringPts[0].y);
                    for (var ri = 1; ri < ringPts.length; ri += 1) {
                        ctx.lineTo(ringPts[ri].x, ringPts[ri].y);
                    }
                    ctx.closePath();
                    ctx.stroke();
                }
            }
        }

        // Depth-sort actors far-to-near and render solid low-poly 3D vehicles & grounded VRUs
        var renderActors = [];
        for (var vIdx = 0; vIdx < vehicles.length; vIdx += 1) {
            renderActors.push({ kind: "veh", item: vehicles[vIdx], yDepth: vehicles[vIdx].y || 0 });
        }
        for (var uIdx = 0; uIdx < vrus.length; uIdx += 1) {
            renderActors.push({ kind: "vru", item: vrus[uIdx], yDepth: vrus[uIdx].y || 0 });
        }
        renderActors.sort(function (a, bObj) { return bObj.yDepth - a.yDepth; });

        var placedLabels = [];
        var obstacleBoxes = [];
        var safeRect = { minX: 6, minY: 30, maxX: w - 6, maxY: h - 6 };

        // Reserve top HUD pill rectangles upfront so actor pills never collide with HUD overlays
        var camTitle = st.cameraLabel
            ? (w < 410 && st.cameraShortLabel ? st.cameraShortLabel : st.cameraLabel)
            : (w < 410 ? "CAM-01 · 3D 透视仿真" : "CAM-01 · 3D 透视仿真视角 · 标定 6.0m x 15.0m");
        var maxRisk = st.maxRisk || 0;
        var riskTxt = maxRisk === 3 ? "L3 CRITICAL" : (maxRisk === 2 ? "L2 CAUTION" : (maxRisk === 1 ? "L1 SPEEDING" : "L0 SAFE"));
        var riskCol = riskColor(pal, maxRisk);

        ctx.save();
        ctx.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        var camTitleW = ctx.measureText(camTitle).width + 12;
        var pillW = ctx.measureText(riskTxt).width + 12;
        ctx.restore();
        placedLabels.push({ x: 10, y: 8, w: camTitleW, h: 18 });
        placedLabels.push({ x: Math.max(10, w - pillW - 10), y: 8, w: pillW, h: 18 });

        var pendingActorPills = [];
        for (var ai2 = 0; ai2 < renderActors.length; ai2 += 1) {
            var entry = renderActors[ai2];
            if (entry.kind === "veh") {
                var v = entry.item;
                var vColor = v.inParkingZone
                    ? pal.textMuted
                    : (v.riskLevel >= 1 ? riskColor(pal, v.riskLevel) : pal.safe);
                var vBox = drawLowPolyVehicle(v, vColor);
                if (!vBox) continue;

                drawCornerBracketBox(ctx, vBox.x, vBox.y, vBox.w, vBox.h, vColor, v.riskLevel >= 2 ? 2.0 : 1.5);
                obstacleBoxes.push(vBox);

                var anchorPt = p3(v.x, v.y, 0);
                if (anchorPt) {
                    ctx.fillStyle = pal.accent;
                    ctx.beginPath();
                    ctx.arc(anchorPt.x, anchorPt.y, 2.8, 0, Math.PI * 2);
                    ctx.fill();
                }

                var spdTxt = isFiniteNum(v.speedKmh) ? v.speedKmh.toFixed(1) + " km/h" : "0.0 km/h";
                var statusSuffix = v.riskLevel >= 3 ? " [ALERT]" : (v.riskLevel === 2 ? " [CAUTION]" : (v.riskLevel === 1 ? " [SPEEDING]" : ""));
                var vLabel = "V-" + padId(v.trackId) + " | " + spdTxt + statusSuffix;
                pendingActorPills.push({
                    priority: 10 + (v.riskLevel || 0),
                    target: { cx: vBox.x + vBox.w * 0.5, cy: vBox.y - 10, rx: vBox.w * 0.5, ry: 2 },
                    text: vLabel,
                    bg: v.riskLevel >= 3 ? pal.critical : pal.bgCard,
                    border: vColor,
                    fg: v.riskLevel >= 3 ? "#ffffff" : pal.textPrimary
                });
            } else {
                var u = entry.item;
                var isBike = u.kind === "bicycle" || u.category === "bicycle" || u.kind === "motorcycle";
                var uColor = u.riskLevel >= 2 ? riskColor(pal, u.riskLevel) : (isBike ? pal.accent : pal.safe);
                var uBox = drawGroundedVruSilhouette(u, uColor, isBike);
                if (!uBox) continue;

                drawCornerBracketBox(ctx, uBox.x, uBox.y, uBox.w, uBox.h, uColor, u.riskLevel >= 2 ? 1.8 : 1.4);
                obstacleBoxes.push(uBox);

                var uAnchor = p3(u.x, u.y, 0);
                if (uAnchor) {
                    ctx.fillStyle = pal.accent;
                    ctx.beginPath();
                    ctx.arc(uAnchor.x, uAnchor.y, 2.5, 0, Math.PI * 2);
                    ctx.fill();
                }

                var uTag = (isBike ? "B-" : "P-") + padId(u.trackId);
                if (isBike && Math.hypot(u.vx || 0, u.vy || 0) > 0.2) {
                    uTag += " " + (Math.hypot(u.vx || 0, u.vy || 0) * 3.6).toFixed(1) + "km/h";
                }
                pendingActorPills.push({
                    priority: 20 + (u.riskLevel || 0),
                    target: { cx: uBox.x + uBox.w * 0.5, cy: uBox.y - 10, rx: uBox.w * 0.5, ry: 2 },
                    text: uTag,
                    bg: u.riskLevel >= 3 ? pal.critical : pal.bgCard,
                    border: uColor,
                    fg: u.riskLevel >= 3 ? "#ffffff" : pal.textPrimary
                });
            }
        }

        // Place actor labels avoiding all projected vehicle & VRU boxes (VRUs & higher risk first)
        pendingActorPills.sort(function (a, bObj) { return bObj.priority - a.priority; });
        for (var pp = 0; pp < pendingActorPills.length; pp += 1) {
            var pItem = pendingActorPills[pp];
            placePill(
                ctx,
                pItem.target,
                pItem.text,
                pItem.bg,
                pItem.border,
                pItem.fg,
                safeRect,
                placedLabels,
                obstacleBoxes
            );
        }

        // Top overlay status bar (clean corner pills with dynamically measured risk pill width)
        drawPill(ctx, 10, 8, camTitle, pal.bgCard, pal.hairline, pal.textSecondary, w);
        drawPill(
            ctx,
            Math.max(10, w - pillW - 10),
            8,
            riskTxt,
            maxRisk === 3 ? pal.critical : pal.bgCard,
            riskCol,
            maxRisk === 3 ? "#ffffff" : riskCol,
            w
        );

        ctx.restore();
    };

    SimulatedCameraCanvas.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        detachResizeObserver(this);
        this._unobserveTheme();
        if (this._probe && this._probe.parentNode) {
            this._probe.parentNode.removeChild(this._probe);
        }
    };

    /* ======================================================================
     * Public global export
     * ====================================================================== */
    global.CampusMotion = {
        RollingCounter: RollingCounter,
        AnimatedSignalPipeline: AnimatedSignalPipeline,
        CampusRoadTwinCanvas: CampusRoadTwinCanvas,
        SpeedTimelineChart: SpeedTimelineChart,
        CalibrationCanvas: CalibrationCanvas,
        AlertAudioSynth: AlertAudioSynth,
        SimulatedCameraCanvas: SimulatedCameraCanvas,
        _buildScenarioState: buildScenarioState,
        _computeCpaAndTtc: computeCpaAndTtc,
        _computeCvmCircleTtc: computeCvmCircleTtc,
        _rolloutVehicleTrajectory: rolloutVehicleTrajectory,
        _predictCtrvArc: predictCtrvArc,
        _SCENARIO_CYCLE_S: SCENARIO_CYCLE_S
    };
})(typeof window !== "undefined" ? window : globalThis);
