-- Enforce FaceTime per-app audio routes pre-dial (2026-09-07, positional).
-- Video menu FLAT layout (verified by index probe):
--   11 Microphone header | 12 Use System | 13 BH16 | 14 BH2 | 15 MacBook mic
--   17 Output header     | 18 Use System | 19 BH16 | 20 BH2 | 21 MacBook spk
-- FaceTime relaunches fresh per dial and its routes drift; agent TTS renders
-- into BH2 (mic must be BH2) and caller audio must arrive on BH16 (output).
on run
	tell application "FaceTime" to activate
	delay 1.2
	set report to ""
	tell application "System Events"
		tell process "FaceTime"
			set vidMenu to menu "Video" of menu bar 1
			-- dynamic layout check: find the Microphone header index
			set micHeaderIdx to 0
			set idx to 0
			repeat with mi in menu items of vidMenu
				set idx to idx + 1
				set n to name of mi as string
				if n is "Microphone" then
					set micHeaderIdx to idx
					exit repeat
				end if
			end repeat
			if micHeaderIdx is 0 then return "ERROR: no Microphone header"
			set micBH2Idx to micHeaderIdx + 3      -- 12,13,14 -> BH2
			set outBH16Idx to micHeaderIdx + 8     -- 18,19 -> BH16
			-- sanity: names at computed positions
			set nMicBH2 to name of menu item micBH2Idx of vidMenu as string
			set nOutBH16 to name of menu item outBH16Idx of vidMenu as string
			if nMicBH2 is not "BlackHole 2ch" then return "ERROR: layout drift at mic BH2 (got " & nMicBH2 & ")"
			if nOutBH16 is not "BlackHole 16ch" then return "ERROR: layout drift at out BH16 (got " & nOutBH16 & ")"
			-- MIC
			set mark1 to ""
			try
				set mark1 to value of attribute "AXMenuItemMarkChar" of menu item micBH2Idx of vidMenu
			end try
			if mark1 is missing value or mark1 is "-" or mark1 is "" then
				click menu item micBH2Idx of vidMenu
				set report to "MIC-SET->BH2; "
			else
				set report to "MIC-OK; "
			end if
			-- OUTPUT
			set mark2 to ""
			try
				set mark2 to value of attribute "AXMenuItemMarkChar" of menu item outBH16Idx of vidMenu
			end try
			if mark2 is missing value or mark2 is "-" or mark2 is "" then
				click menu item outBH16Idx of vidMenu
				set report to report & "OUT-SET->BH16"
			else
				set report to report & "OUT-OK"
			end if
		end tell
	end tell
	return report
end run