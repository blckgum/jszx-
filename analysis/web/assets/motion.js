/* Campus Conflict Perception Console - Motion & Visualization Engine
 * Pure classic browser script (no build step, no external dependencies).
 * Exports window.CampusMotion = {
 *   RollingCounter,
 *   AnimatedSignalPipeline,
 *   CampusRoadTwinCanvas,
 *   SpeedTimelineChart
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

    /* ======================================================================
     * 1. RollingCounter (ported from number-flow)
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
        this._lastDigits = [];
        this._lastNumText = null;
        this._lastValid = null;
        this._lastUnit = null;
        this._lastUpdateMs = 0;
        this._lastDurationMs = 320;
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
        var strips = this._el.querySelectorAll(".nf-strip");
        var dur = this._lastDurationMs || 320;
        for (var i = 0; i < strips.length; i += 1) {
            strips[i].style.transition = isReduced
                ? "none"
                : "transform " + dur + "ms cubic-bezier(0.22, 1, 0.36, 1)";
        }
    };

    RollingCounter.prototype._createDigitSlot = function (initialDigit, targetDigit, isReduced) {
        var slot = document.createElement("span");
        slot.className = "nf-slot";
        slot.setAttribute("aria-hidden", "true");
        slot.style.display = "inline-block";
        slot.style.overflow = "hidden";
        slot.style.height = "1.28em";
        slot.style.lineHeight = "1.28";
        slot.style.verticalAlign = "text-bottom";
        slot.style.width = "1ch";
        slot.style.textAlign = "center";

        var strip = document.createElement("span");
        strip.className = "nf-strip";
        strip.style.display = "flex";
        strip.style.flexDirection = "column";
        strip.style.lineHeight = "1.28";
        strip.style.fontVariantNumeric = "tabular-nums";
        strip.style.willChange = isReduced ? "auto" : "transform";

        for (var d = 0; d <= 9; d += 1) {
            var cell = document.createElement("span");
            cell.className = "nf-digit";
            cell.textContent = String(d);
            cell.style.display = "flex";
            cell.style.alignItems = "center";
            cell.style.justifyContent = "center";
            cell.style.height = "1.28em";
            cell.style.lineHeight = "1.28";
            strip.appendChild(cell);
        }

        if (isReduced) {
            strip.style.transition = "none";
            strip.style.transform = "translateY(-" + (targetDigit * 10) + "%)";
        } else {
            strip.style.transition = "none";
            strip.style.transform = "translateY(-" + (initialDigit * 10) + "%)";
        }

        slot.appendChild(strip);
        return { slot: slot, strip: strip, targetDigit: targetDigit };
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

        var nowMs = typeof performance !== "undefined" ? performance.now() : Date.now();
        var elapsedMs = this._lastUpdateMs > 0 ? (nowMs - this._lastUpdateMs) : 9999;
        this._lastUpdateMs = nowMs;
        var animDurationMs = (decimals === 0 && elapsedMs < 350) ? 180 : 320;
        this._lastDurationMs = animDurationMs;
        this._lastNumText = numText;
        this._lastValid = isValid;
        this._lastUnit = unit;

        var fullAria = isValid && unit ? (numText + " " + unit) : numText;
        this._el.setAttribute("aria-label", fullAria);
        this._el.setAttribute("data-value", numText);

        var isReduced = this._motionTracker.isReduced();
        var prevDigits = this._lastDigits;
        var newDigits = [];
        var digitSlots = [];

        if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
            global.cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }

        while (this._el.firstChild) {
            this._el.removeChild(this._el.firstChild);
        }

        if (!isValid) {
            var fbSpan = document.createElement("span");
            fbSpan.className = "nf-char nf-fallback";
            fbSpan.setAttribute("aria-hidden", "true");
            fbSpan.textContent = fallback;
            this._el.appendChild(fbSpan);
            this._lastDigits = [];
            return;
        }

        // Count digit characters first to index right-to-left for smooth carry continuity
        var totalDigits = 0;
        for (var c = 0; c < numText.length; c += 1) {
            var ch = numText.charCodeAt(c);
            if (ch >= 48 && ch <= 57) totalDigits += 1;
        }

        var digitCursor = 0;
        for (var i = 0; i < numText.length; i += 1) {
            var chStr = numText.charAt(i);
            var code = numText.charCodeAt(i);
            if (code >= 48 && code <= 57) {
                var digit = code - 48;
                var fromRight = totalDigits - 1 - digitCursor;
                var prevIdx = prevDigits.length - 1 - fromRight;
                var startDigit = (prevIdx >= 0 && prevIdx < prevDigits.length) ? prevDigits[prevIdx] : digit;
                var created = this._createDigitSlot(startDigit, digit, isReduced);
                this._el.appendChild(created.slot);
                digitSlots.push(created);
                newDigits.push(digit);
                digitCursor += 1;
            } else {
                var sep = document.createElement("span");
                sep.className = "nf-char";
                sep.setAttribute("aria-hidden", "true");
                sep.textContent = chStr;
                sep.style.display = "inline-block";
                sep.style.lineHeight = "1.28";
                sep.style.verticalAlign = "text-bottom";
                this._el.appendChild(sep);
            }
        }

        if (unit) {
            var unitSpan = document.createElement("span");
            unitSpan.className = "nf-unit";
            unitSpan.setAttribute("aria-hidden", "true");
            unitSpan.textContent = unit;
            unitSpan.style.lineHeight = "1.28";
            this._el.appendChild(unitSpan);
        }

        this._lastDigits = newDigits;

        if (!isReduced && digitSlots.length > 0 && typeof global.requestAnimationFrame === "function") {
            var self = this;
            this._rafId = global.requestAnimationFrame(function () {
                self._rafId = null;
                if (self._destroyed) return;
                for (var k = 0; k < digitSlots.length; k += 1) {
                    var item = digitSlots[k];
                    item.strip.style.transition = "transform " + animDurationMs + "ms cubic-bezier(0.22, 1, 0.36, 1)";
                    item.strip.style.transform = "translateY(-" + (item.targetDigit * 10) + "%)";
                }
            });
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

        if (typeof ResizeObserver === "function") {
            this._resizeObserver = new ResizeObserver(function () {
                self._measurePaths();
            });
            this._resizeObserver.observe(this._svg);
        } else {
            this._resizeObserver = null;
        }

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
        var alertBadge = this._svg.querySelector("[data-stage='alert'] rect, [data-node='alert'] rect");
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
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
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

    function computeCpaAndTtc(veh, vru, horizonS) {
        var horizon = horizonS || 3.5;
        var dt = 0.05;
        var steps = Math.round(horizon / dt);

        var psi = veh.headingRad;
        var x = veh.x;
        var y = veh.y;
        var v = Math.hypot(veh.vx, veh.vy);
        var a = veh.accelMs2 || 0;
        var omega = veh.yawRate || 0;

        var px = vru.x;
        var py = vru.y;
        var pvx = vru.vx;
        var pvy = vru.vy;

        var halfL = (veh.lengthM || 4.5) * 0.5;
        var halfW = (veh.widthM || 1.9) * 0.5;
        var safeR = vru.radiusM || 0.42;

        var currentDist = Math.hypot(px - x, py - y);
        var minClearance = Infinity;
        var bestX = x;
        var bestY = y;
        var ttc = null;

        // If vehicle is effectively stationary and not overlapping right now, no TTC conflict
        if (v < 0.25 && a <= 0.1) {
            var dx0 = px - x;
            var dy0 = py - y;
            var c0 = Math.cos(psi);
            var s0 = Math.sin(psi);
            var lx0 = dx0 * c0 + dy0 * s0;
            var ly0 = -dx0 * s0 + dy0 * c0;
            var cx0 = clamp(lx0, -halfL, halfL);
            var cy0 = clamp(ly0, -halfW, halfW);
            var clear0 = Math.hypot(lx0 - cx0, ly0 - cy0) - safeR;
            return {
                currentDist: currentDist,
                minDistanceM: Math.max(0, clear0),
                ttcSeconds: clear0 <= 0.05 ? 0 : null,
                conflictGroundXY: [x + cx0 * c0 - cy0 * s0, y + cx0 * s0 + cy0 * c0]
            };
        }

        // Second-order deceleration rollout factor for realistic MTTC/TTC transition
        var effA = a < 0 ? a * 0.72 : a;

        for (var i = 0; i <= steps; i += 1) {
            var t = i * dt;
            var dx = (px + pvx * t) - x;
            var dy = (py + pvy * t) - y;
            var cosP = Math.cos(psi);
            var sinP = Math.sin(psi);
            var lx = dx * cosP + dy * sinP;
            var ly = -dx * sinP + dy * cosP;
            var cx = clamp(lx, -halfL, halfL);
            var cy = clamp(ly, -halfW, halfW);
            var clearance = Math.hypot(lx - cx, ly - cy) - safeR;

            if (clearance < minClearance) {
                minClearance = clearance;
                bestX = x + (cx * cosP - cy * sinP + dx) * 0.5;
                bestY = y + (cx * sinP + cy * cosP + dy) * 0.5;
            }
            if (ttc === null && clearance <= 0.12) {
                ttc = t;
            }

            // Advance 2nd-order CTRV rollout step
            var nextV = Math.max(0, v + effA * dt);
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
        }

        return {
            currentDist: currentDist,
            minDistanceM: Math.max(0, minClearance),
            ttcSeconds: ttc,
            conflictGroundXY: [bestX, bestY]
        };
    }

    function predictCtrvArc(veh, horizonS, steps) {
        var pts = [{ x: veh.x, y: veh.y, t: 0 }];
        var count = steps || 15;
        var dt = (horizonS || 2.5) / count;
        var x = veh.x;
        var y = veh.y;
        var psi = veh.headingRad;
        var v = Math.hypot(veh.vx, veh.vy);
        var a = veh.accelMs2 || 0;
        var omega = veh.yawRate || 0;

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
            pts.push({ x: x, y: y, t: i * dt });
        }
        return pts;
    }

    function buildScenarioState(scenarioId, cycleTimeS) {
        var t = cycleTimeS % 10.0;
        var vehicles = [];
        var vrus = [];
        var parkingZones = [
            [[1.85, 2.2], [3.05, 2.2], [3.05, 6.2], [1.85, 6.2]]
        ];

        if (scenarioId === "cyclist_vru_cutin") {
            // Scenario 2: Cyclist blind-spot cut-in past parked car in exclusion zone
            var suvX = 0.25;
            var suvY = 0.5;
            var suvSpd = 6.2; // 22.3 km/h cruising
            var suvAccel = 0;
            var suvYaw = 0;

            if (t < 1.2) {
                suvY = 0.5 + t * 0.75;
                suvSpd = 6.2;
            } else if (t < 2.8) {
                var p1 = (t - 1.2) / 1.6;
                suvY = 1.40 + p1 * 1.35;
                suvSpd = 6.2 - p1 * 3.4; // smooth 22.3 -> 10.1 km/h
                suvAccel = -2.1;
            } else if (t < 4.9) {
                var p2 = (t - 2.8) / 2.1;
                suvY = 2.75 + p2 * 1.75;
                suvX = 0.25 - p2 * 0.30;
                suvSpd = Math.max(0.2, 2.8 - p2 * 2.6); // 10.1 -> 0.7 km/h
                suvAccel = -1.4;
                suvYaw = 0.16;
            } else if (t < 6.8) {
                suvY = 4.50;
                suvX = -0.05;
                suvSpd = 0.0;
                suvAccel = 0;
                suvYaw = 0.04;
            } else {
                var p3 = (t - 6.8) / 3.2;
                suvY = 4.50 + p3 * 6.0;
                suvX = -0.05 + p3 * 0.20;
                suvSpd = Math.min(5.2, p3 * 5.6);
                suvAccel = 1.6;
                suvYaw = -0.05;
            }
            var suvPsi = Math.PI * 0.5 + suvYaw * 0.6;
            var suvVx = Math.cos(suvPsi) * suvSpd;
            var suvVy = Math.sin(suvPsi) * suvSpd;

            vehicles.push({
                trackId: 1,
                kind: "suv",
                x: suvX,
                y: suvY,
                vx: suvVx,
                vy: suvVy,
                ax: 0,
                ay: suvAccel,
                accelMs2: suvAccel,
                headingRad: suvPsi,
                yawRate: suvYaw,
                lengthM: 4.7,
                widthM: 1.95,
                speedKmh: suvSpd * 3.6,
                inParkingZone: false
            });

            vehicles.push({
                trackId: 2,
                kind: "car",
                x: 2.42,
                y: 4.2,
                vx: 0,
                vy: 0,
                ax: 0,
                ay: 0,
                accelMs2: 0,
                headingRad: Math.PI * 0.5,
                yawRate: 0,
                lengthM: 4.4,
                widthM: 1.85,
                speedKmh: 0,
                inParkingZone: true
            });

            // Cyclist emerges from behind V-02 laterally across SUV lane at y~7.4..8.6m
            var cycX = 2.15;
            var cycY = 7.0;
            var cycVx = 0;
            var cycVy = 0;
            if (t >= 1.1 && t < 6.8) {
                var cp = (t - 1.1) / 5.7;
                cycX = 2.15 - cp * 4.75;
                cycY = 7.0 + cp * 1.9;
                cycVx = -1.05;
                cycVy = 0.35;
            } else if (t >= 6.8) {
                cycX = -2.60;
                cycY = 8.9 + (t - 6.8) * 0.9;
                cycVx = 0;
                cycVy = 0.9;
            }

            vrus.push({
                trackId: 12,
                kind: "bicycle",
                category: "bicycle",
                x: cycX,
                y: cycY,
                vx: cycVx,
                vy: cycVy,
                radiusM: 0.50,
                lengthM: 1.8,
                widthM: 0.7,
                headingRad: Math.atan2(cycVy || 0.4, cycVx || -0.3)
            });

            vrus.push({
                trackId: 5,
                kind: "pedestrian",
                category: "pedestrian",
                x: -2.75,
                y: 10.6 + Math.sin(t * 0.6) * 0.4,
                vx: 0,
                vy: 0.35,
                radiusM: 0.40
            });
        } else if (scenarioId === "ctrv_curve_yield") {
            // Scenario 3: Turning vehicle with non-zero yaw rate & curved CTRV trajectory
            var cx = 1.45;
            var cy = 0.6;
            var cSpd = 6.8;
            var cAccel = 0;
            var cYawRate = 0.24;
            var cHeading = Math.PI * 0.5 + 0.12;

            if (t < 1.2) {
                cy = 0.6 + t * 0.80;
                cx = 1.45 - t * 0.14;
                cSpd = 6.8;
                cYawRate = 0.24;
                cHeading = Math.PI * 0.5 + 0.12 + t * 0.08;
            } else if (t < 2.8) {
                var tp1 = (t - 1.2) / 1.6;
                cy = 1.56 + tp1 * 1.45;
                cx = 1.28 - tp1 * 0.42;
                cSpd = 5.4 - tp1 * 2.2; // 19.4 -> 11.5 km/h
                cAccel = -1.1;
                cYawRate = 0.30;
                cHeading = Math.PI * 0.5 + 0.22 + tp1 * 0.14;
            } else if (t < 4.8) {
                var tp2 = (t - 2.8) / 2.0;
                cy = 3.01 + tp2 * 1.85;
                cx = 0.86 - tp2 * 0.52;
                cSpd = Math.max(0.2, 3.2 - tp2 * 3.0);
                cAccel = -1.2;
                cYawRate = 0.28;
                cHeading = Math.PI * 0.5 + 0.36 + tp2 * 0.12;
            } else if (t < 7.0) {
                cy = 4.86;
                cx = 0.34;
                cSpd = 0.0;
                cAccel = 0;
                cYawRate = 0.0;
                cHeading = Math.PI * 0.5 + 0.48;
            } else {
                var tp3 = (t - 7.0) / 3.0;
                cy = 4.86 + tp3 * 4.8;
                cx = 0.34 - tp3 * 1.45;
                cSpd = Math.min(5.0, tp3 * 5.2);
                cAccel = 1.6;
                cYawRate = -0.10;
                cHeading = Math.PI * 0.5 + 0.48 - tp3 * 0.18;
            }

            vehicles.push({
                trackId: 1,
                kind: "car",
                x: cx,
                y: cy,
                vx: Math.cos(cHeading) * cSpd,
                vy: Math.sin(cHeading) * cSpd,
                ax: 0,
                ay: cAccel,
                accelMs2: cAccel,
                headingRad: cHeading,
                yawRate: cYawRate,
                lengthM: 4.5,
                widthM: 1.9,
                speedKmh: cSpd * 3.6,
                inParkingZone: false
            });

            // Pedestrian P-04 crosses right-to-left into the curved CTRV turn path at y=8.15m
            var pedX = 1.95;
            var pedVx = -0.4;
            if (t >= 0.8 && t < 6.8) {
                var pp = (t - 0.8) / 6.0;
                pedX = 1.95 - pp * 5.05; // exits safely to x = -3.10 sidewalk
                pedVx = -0.82;
            } else if (t >= 6.8) {
                pedX = -3.10;
                pedVx = 0;
            }
            vrus.push({
                trackId: 4,
                kind: "pedestrian",
                category: "pedestrian",
                x: pedX,
                y: 8.15,
                vx: pedVx,
                vy: 0,
                radiusM: 0.42
            });
        } else {
            // Scenario 1 (default): Crosswalk emergency braking (34.2 km/h L1 -> L2 -> L3 -> yield before crosswalk)
            var vY = 0.5;
            var vSpeedMs = 9.5; // 34.2 km/h (> 30 km/h L1)
            var vAccel = 0;

            if (t < 1.1) {
                vY = 0.5 + t * 0.70;
                vSpeedMs = 9.5;
                vAccel = 0;
            } else if (t < 2.3) {
                var b1 = (t - 1.1) / 1.2;
                vY = 1.27 + b1 * 1.15;
                vSpeedMs = 4.6 - b1 * 1.4; // braking into L2 transition (~16.5 -> 11.5 km/h)
                vAccel = -1.2;
            } else if (t < 4.7) {
                var b2 = (t - 2.3) / 2.4;
                vY = 2.42 + b2 * 2.20;
                vSpeedMs = Math.max(0.2, 3.2 - b2 * 3.0);
                vAccel = -1.4;
            } else if (t < 7.0) {
                vY = 4.62; // Front bumper at 4.62 + 2.25 = 6.87m, safely before 7.25m crosswalk
                vSpeedMs = 0.0;
                vAccel = 0;
            } else {
                var b3 = (t - 7.0) / 3.0;
                vY = 4.62 + b3 * 6.5;
                vSpeedMs = Math.min(5.6, b3 * 6.0);
                vAccel = 1.9;
            }

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

            // Pedestrian P-07 starts on right sidewalk -> steps into crosswalk -> exits left sidewalk cleanly
            var p7x = 2.75;
            var p7vx = -0.45;
            if (t < 1.2) {
                p7x = 2.75 - (t / 1.2) * 0.90;
                p7vx = -0.45;
            } else if (t < 6.7) {
                var cp7 = (t - 1.2) / 5.5;
                p7x = 1.85 - cp7 * 4.70;
                p7vx = -0.95;
            } else {
                p7x = -2.85;
                p7vx = 0;
            }

            vrus.push({
                trackId: 7,
                kind: "pedestrian",
                category: "pedestrian",
                x: p7x,
                y: 7.90,
                vx: p7vx,
                vy: 0,
                radiusM: 0.42
            });

            vrus.push({
                trackId: 3,
                kind: "pedestrian",
                category: "pedestrian",
                x: 2.70,
                y: 11.2 - Math.sin(t * 0.4) * 0.5,
                vx: 0,
                vy: -0.3,
                radiusM: 0.40
            });
        }

        return {
            bounds: { xMin: -3.2, xMax: 3.2, yMin: 0.0, yMax: 15.0 },
            parkingZones: parkingZones,
            vehicles: vehicles,
            vrus: vrus
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
        this._speedLimitKmh = toNum(opts.speedLimitKmh, 30.0);

        this._mode = "simulation";
        this._scenarioId = "crosswalk_braking";
        this._userPaused = false;
        this._offscreenPaused = false;
        this._destroyed = false;

        this._simTimeS = 3.8; // Start near meaningful visual peak if reduced motion or paused
        this._simFrameCount = 0;
        this._lastDrawTs = null;
        this._lastTelemetryTs = 0;
        this._rafId = null;
        this._bgTimerId = null;
        this._lastBgTs = null;

        // 0.5m PET spatial occupancy grid: key -> { vehId, vehTime, vruId, vruTime, petS }
        this._petCells = Object.create(null);
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

        if (typeof ResizeObserver === "function") {
            this._resizeObserver = new ResizeObserver(function () {
                self._measureCanvas();
                self._renderCurrent();
            });
            this._resizeObserver.observe(this._canvas);
        } else {
            this._resizeObserver = null;
        }

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
        // Seed recent swept-OBB history behind initial keyframe so PET grid is intelligible immediately
        var startT = Math.max(0, this._simTimeS - 2.0);
        for (var t = startT; t <= this._simTimeS; t += 0.25) {
            var st = buildScenarioState(this._scenarioId, t);
            this._recordPetGrid(st.vehicles, st.vrus, t);
        }
    };

    CampusRoadTwinCanvas.prototype._recordPetGrid = function (vehicles, vrus, timeS) {
        var cellSize = 0.5;
        for (var vi = 0; vi < vehicles.length; vi += 1) {
            var v = vehicles[vi];
            if (v.inParkingZone || (v.speedKmh !== undefined && v.speedKmh < 0.4)) continue;
            var halfL = (v.lengthM || 4.5) * 0.5;
            var halfW = (v.widthM || 1.9) * 0.5;
            var cosP = Math.cos(v.headingRad || 0);
            var sinP = Math.sin(v.headingRad || 0);

            for (var lx = -halfL; lx <= halfL + 0.01; lx += cellSize) {
                for (var ly = -halfW; ly <= halfW + 0.01; ly += cellSize) {
                    var gx = v.x + lx * cosP - ly * sinP;
                    var gy = v.y + lx * sinP + ly * cosP;
                    var cx = Math.floor(gx / cellSize);
                    var cy = Math.floor(gy / cellSize);
                    var key = cx + "," + cy;
                    var existing = this._petCells[key];
                    if (!existing) {
                        this._petCells[key] = {
                            cx: cx,
                            cy: cy,
                            vehId: v.trackId,
                            vehTime: timeS,
                            vruId: null,
                            vruTime: -999,
                            petS: null
                        };
                    } else {
                        existing.vehId = v.trackId;
                        existing.vehTime = timeS;
                        if (existing.vruId !== null && timeS >= existing.vruTime && (timeS - existing.vruTime) <= 2.2) {
                            existing.petS = timeS - existing.vruTime;
                        }
                    }
                }
            }
        }

        for (var pi = 0; pi < vrus.length; pi += 1) {
            var p = vrus[pi];
            var pcx = Math.floor(p.x / cellSize);
            var pcy = Math.floor(p.y / cellSize);
            var pkey = pcx + "," + pcy;
            var cell = this._petCells[pkey];
            if (cell && cell.vehTime <= timeS && (timeS - cell.vehTime) <= 2.4) {
                cell.vruId = p.trackId;
                cell.vruTime = timeS;
                cell.petS = Math.max(0.05, timeS - cell.vehTime);
            }
        }

        // Prune cells older than 3.6s to keep map clean and bounded
        var keys = Object.keys(this._petCells);
        for (var k = 0; k < keys.length; k += 1) {
            var rec = this._petCells[keys[k]];
            if (timeS - rec.vehTime > 3.6) {
                delete this._petCells[keys[k]];
            }
        }
    };

    CampusRoadTwinCanvas.prototype._evaluateSimulation = function (simData, timeS) {
        var conflicts = [];
        var driverAlerts = [];
        var pedestrianAlerts = [];
        var vehMap = Object.create(null);
        var maxRisk = 0;
        var maxSpeed = 0;
        var minTtc = null;

        this._recordPetGrid(simData.vehicles, simData.vrus, timeS);

        // Find active PET overlaps per pair
        var pairPet = Object.create(null);
        var cellKeys = Object.keys(this._petCells);
        for (var c = 0; c < cellKeys.length; c += 1) {
            var cell = this._petCells[cellKeys[c]];
            if (cell.petS !== null && timeS - cell.vruTime <= 1.8) {
                var pk = cell.vehId + ":" + cell.vruId;
                if (pairPet[pk] === undefined || cell.petS < pairPet[pk]) {
                    pairPet[pk] = cell.petS;
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
            for (var pi = 0; pi < simData.vrus.length; pi += 1) {
                var vru = simData.vrus[pi];
                var res = computeCpaAndTtc(veh, vru, 3.5);
                var pk2 = veh.trackId + ":" + vru.trackId;
                var petS = pairPet[pk2] !== undefined ? pairPet[pk2] : null;
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
                var targetLabel = bestConflict.targetKind === "bicycle" ? "骑行者" : "行人";
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

        this._latestSimSnapshot = {
            timestampS: Number(timeS.toFixed(2)),
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
                t: Number(timeS.toFixed(2)),
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
            maxRisk: maxRisk
        };
    };

    CampusRoadTwinCanvas.prototype._scenarioTitle = function (id) {
        if (id === "cyclist_vru_cutin") return "SIM · 骑行者盲区切入";
        if (id === "ctrv_curve_yield") return "SIM · 弯道 CTRV 轨迹会车";
        return "SIM · 斑马线横穿急刹";
    };

    CampusRoadTwinCanvas.prototype._stepSimulation = function (dt, forceEmit, skipRender) {
        if (this._mode !== "simulation") return;
        var prevMod = this._simTimeS % 10.0;
        this._simTimeS += dt;
        var nextMod = this._simTimeS % 10.0;
        if (nextMod < prevMod) {
            // Reset PET grid smoothly on scenario cycle restart
            this._petCells = Object.create(null);
        }
        this._simFrameCount += 1;
        var simData = buildScenarioState(this._scenarioId, this._simTimeS);
        this._evaluateSimulation(simData, this._simTimeS);
        if (!skipRender) {
            this._renderCurrent();
        }
        this._emitSimTelemetry(Boolean(forceEmit));
    };

    CampusRoadTwinCanvas.prototype._emitSimTelemetry = function (force) {
        if (!this._onTelemetry || !this._latestSimSnapshot || this._mode !== "simulation") {
            return;
        }
        var now = typeof performance !== "undefined" ? performance.now() : Date.now();
        if (force || (now - this._lastTelemetryTs >= 110)) {
            this._lastTelemetryTs = now;
            this._onTelemetry(this._latestSimSnapshot);
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
        for (var gx = firstX; gx <= Math.floor(b.xMax); gx += 1) {
            var px1 = toPx(gx, b.yMin);
            var px2 = toPx(gx, b.yMax);
            ctx.strokeStyle = pal.gridLine;
            ctx.beginPath();
            ctx.moveTo(px1.x, px1.y);
            ctx.lineTo(px2.x, px2.y);
            ctx.stroke();

            if (gx % 2 === 0 || spanX <= 7) {
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

        // Crosswalk zebra stripes around y = 7.8m (plus subtle sidewalk curb ramps)
        if (7.8 >= b.yMin + 0.5 && 7.8 <= b.yMax - 0.5) {
            var cwLeft = b.xMin + 0.35;
            var cwRight = b.xMax - 0.35;
            var cwY0 = 7.25;
            var cwY1 = 8.45;
            var cwTopPx = toPx(b.xMin, cwY1).y;
            var cwBotPx = toPx(b.xMin, cwY0).y;
            if (leftSidewalkW > 12) {
                ctx.fillStyle = withAlpha(pal.textPrimary, pal.isLight ? 0.045 : 0.035);
                ctx.fillRect(Math.max(padL, roadTopLeft.x - 1.2 * scale), cwTopPx, Math.min(leftSidewalkW, 1.2 * scale), cwBotPx - cwTopPx);
            }
            if (rightSidewalkW > 12) {
                ctx.fillStyle = withAlpha(pal.textPrimary, pal.isLight ? 0.045 : 0.035);
                ctx.fillRect(roadBotRight.x, cwTopPx, Math.min(rightSidewalkW, 1.2 * scale), cwBotPx - cwTopPx);
            }
            ctx.fillStyle = withAlpha(pal.textPrimary, pal.isLight ? 0.14 : 0.11);
            for (var sx = cwLeft; sx < cwRight - 0.25; sx += 0.65) {
                var spA = toPx(sx, cwY1);
                var spB = toPx(Math.min(sx + 0.34, cwRight), cwY0);
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
            var cosA = Math.abs(Math.cos(ov.headingRad || Math.PI * 0.5));
            var sinA = Math.abs(Math.sin(ov.headingRad || Math.PI * 0.5));
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
            var pzW = ctx.measureText(pzText).width + 6;
            var pzH = 14;
            var pzCand = [
                { x: pMaxX - pzW - 2, y: pMinY - pzH - 3 },
                { x: pMinX + 2, y: pMinY - pzH - 3 },
                { x: pMaxX + 6, y: pMinY + 2 },
                { x: pMinX + 3, y: pMaxY - pzH - 2 }
            ];
            var pzPicked = chooseBestBox(pzCand, pzW, pzH, safeArea, placedBoxes, obstacleBoxes);
            ctx.fillStyle = withAlpha(pal.parkingZone, 0.94);
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(pzText, pzPicked.x + 2, pzPicked.y + pzH * 0.5);
            placedBoxes.push(pzPicked);
        }

        // Clip dynamic road overlays (PET grid, CTRV curves, TTC rings) strictly inside road network draw area
        var netClipX = padL;
        var netClipY = Math.max(30, roadTopLeft.y);
        var netClipW = Math.max(40, (w - padR) - padL);
        var netClipH = Math.max(20, roadBotRight.y - netClipY);

        ctx.save();
        ctx.beginPath();
        ctx.rect(netClipX, netClipY, netClipW, netClipH);
        ctx.clip();

        // 0.5m PET swept-OBB occupancy grid
        var refTime = st.simTimeS;
        var cellSize = 0.5;
        var cellKeys = Object.keys(this._petCells);
        for (var ci = 0; ci < cellKeys.length; ci += 1) {
            var cell = this._petCells[cellKeys[ci]];
            var age = refTime - cell.vehTime;
            if (age < 0 || age > 3.6) continue;
            var wx0 = cell.cx * cellSize;
            var wy0 = cell.cy * cellSize;
            if (wx0 + cellSize < b.xMin || wx0 > b.xMax || wy0 + cellSize < b.yMin || wy0 > b.yMax) continue;

            var cTopLeft = toPx(wx0, wy0 + cellSize);
            var cBotRight = toPx(wx0 + cellSize, wy0);
            var cw = cBotRight.x - cTopLeft.x;
            var ch = cBotRight.y - cTopLeft.y;
            var fade = clamp(1 - age / 3.6, 0.12, 1);

            if (cell.petS !== null && (refTime - cell.vruTime) <= 2.2) {
                var petUrgent = cell.petS <= 1.0;
                var petTint = petUrgent ? pal.critical : pal.accent;
                ctx.fillStyle = withAlpha(petTint, 0.24 * fade);
                ctx.fillRect(cTopLeft.x + 0.5, cTopLeft.y + 0.5, cw - 1, ch - 1);
                ctx.strokeStyle = withAlpha(petTint, 0.65 * fade);
                ctx.lineWidth = 1;
                ctx.strokeRect(cTopLeft.x + 0.5, cTopLeft.y + 0.5, cw - 1, ch - 1);
            } else {
                ctx.fillStyle = withAlpha(pal.accent, 0.08 * fade);
                ctx.fillRect(cTopLeft.x + 0.5, cTopLeft.y + 0.5, cw - 1, ch - 1);
            }
        }

        // Draw 2nd-order CTRV/Kalman prediction arcs beneath vehicles (clipped cleanly inside road network)
        for (var vi = 0; vi < st.vehicles.length; vi += 1) {
            var veh = st.vehicles[vi];
            var spdMs = Math.hypot(veh.vx || 0, veh.vy || 0);
            if (spdMs < 0.35 || veh.inParkingZone) continue;

            var arcPts = predictCtrvArc(veh, 2.5, 16);
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

        // Draw rounded OBB vehicles (record pill tasks for collision-free layout)
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

            ctx.save();
            ctx.translate(vPx.x, vPx.y);
            ctx.rotate(-(v.headingRad || Math.PI * 0.5));

            drawRoundedRectPath(ctx, -lenPx * 0.5, -widPx * 0.5, lenPx, widPx, cornerPx);
            ctx.fillStyle = withAlpha(vColor, v.inParkingZone ? 0.14 : 0.20);
            ctx.fill();
            ctx.lineWidth = v.riskLevel >= 2 ? 2.0 : 1.5;
            ctx.strokeStyle = vColor;
            ctx.stroke();

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

            var spdDisplay = isFiniteNum(v.speedKmh) ? v.speedKmh.toFixed(1) + "km/h" : "0.0km/h";
            var badgeText = "V-" + padId(v.trackId) + " " + spdDisplay;
            var cosV = Math.abs(Math.cos(v.headingRad || Math.PI * 0.5));
            var sinV = Math.abs(Math.sin(v.headingRad || Math.PI * 0.5));
            pendingVehiclePills.push({
                cx: vPx.x,
                cy: vPx.y,
                rx: (lenPx * cosV + widPx * sinV) * 0.5,
                ry: (lenPx * sinV + widPx * cosV) * 0.5,
                text: badgeText,
                border: withAlpha(vColor, 0.55)
            });
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
        this._replayPrevTracks = Object.create(null);
        this._lastReplayFrameIdx = -1;
        if (this._mode === "simulation") {
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
        this._simTimeS = (this._userPaused || this._motionTracker.isReduced()) ? 3.8 : 0.2;
        this._seedInitialPetTrail();
        if (this._mode === "simulation") {
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
            this._replayPrevTracks = Object.create(null);
        }
        this._lastReplayFrameIdx = frameIdx;

        var rawVehicles = Array.isArray(fd.vehicles) ? fd.vehicles : [];
        var rawVrus = Array.isArray(fd.vrus) ? fd.vrus : [];
        var rawObjects = Array.isArray(fd.objects) ? fd.objects : (Array.isArray(fd.tracks) ? fd.tracks : []);
        for (var oi = 0; oi < rawObjects.length; oi += 1) {
            var obj = rawObjects[oi];
            var cat = String(obj.category || obj.kind || "").toLowerCase();
            if (cat === "pedestrian" || cat === "person" || cat === "bicycle") {
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

        this._recordPetGrid(normVehicles, normVrus, ts);

        this._currentRenderState = {
            bounds: bounds,
            parkingZones: Array.isArray(fd.parkingZones) ? fd.parkingZones : [],
            vehicles: normVehicles,
            vrus: normVrus,
            conflicts: normConflicts,
            label: fd.label || ("REPLAY · F" + padId(frameIdx) + " · " + ts.toFixed(1) + "s"),
            simTimeS: ts,
            maxRisk: 0
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
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
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

        if (typeof ResizeObserver === "function") {
            this._resizeObserver = new ResizeObserver(function () {
                self._measure();
                self._render();
            });
            this._resizeObserver.observe(this._canvas);
        } else {
            this._resizeObserver = null;
        }

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
        ctx.fillText("RISK", padL - 6, botBandY + botBandH * 0.5);

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
            drawPill(ctx, hx + 8, padT + 2, tipText, pal.bgRaised, pal.hairline, pal.textPrimary, w);
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
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
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
        SpeedTimelineChart: SpeedTimelineChart
    };
})(typeof window !== "undefined" ? window : globalThis);
