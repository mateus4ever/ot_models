# JIRA Survival Strategy
**For Corporate Environments with Enforcers**

**Last Updated:** January 3, 2026

---

## Philosophy

JIRA in corporate = **compliance theater**. Fill required fields with plausible defaults, minimal effort.

**Goal:** Satisfy enforcers in 10 minutes/week, spend rest of time on actual work.

---

## The Three JIRA Tasks

### 1. ANALYZE (Extract Info)
### 2. CREATE (New Epics/Stories)  
### 3. MAINTENANCE (Update/Comment/Log Time)

---

## 1. ANALYZE - Extract Description & Comments

### Export from JIRA to Readable Format

**Step 1: Export CSV**
1. JIRA → Issues → Search (your filter)
2. Columns → Add: Summary, Description, Comment, Epic Link, Status
3. Export → CSV (All fields)
4. Save as `jira_export.csv`

**Step 2: Convert to Markdown (Optional)**

```python
import pandas as pd

# Read JIRA export
df = pd.read_csv('jira_export.csv')

# Generate readable markdown per story
for _, row in df.iterrows():
    filename = f"story_{row['Issue key'].replace('-', '_')}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {row['Summary']}\n\n")
        f.write(f"**Key:** {row['Issue key']}\n")
        f.write(f"**Epic:** {row['Epic Link']}\n")
        f.write(f"**Status:** {row['Status']}\n\n")
        f.write(f"## Description\n{row['Description']}\n\n")
        f.write(f"## Comments\n{row['Comment']}\n\n")
```

**Result:** Clean markdown files you can read/search locally.

---

## 2. CREATE - New Epics/Stories

### Strategy: Minimal Effort, Maximum Compliance

#### Option A: Clone Template Story

**Setup (Once):**

1. Create ONE template story: `PROJ-TEMPLATE`
2. Fill with defaults:
   - Summary: `[TEMPLATE] Story Title`
   - Description: `See detailed specs: [link]`
   - Due Date: (Leave blank initially)
   - Assignee: Unassigned
   - Estimate: 4h
   - Label: `backend`
   - Release: `v.Next`

**Every New Story:**
1. Find `PROJ-TEMPLATE`
2. Click "Clone"
3. Change ONLY: Summary, Description, Epic Link
4. Save

**Time:** 30 seconds per story

---

#### Option B: Bulk Create via CSV

**If CSV import is allowed:**

**Create CSV with required + bullshit fields:**
```csv
Summary,Description,Issue Type,Epic Link,Due Date,Original Estimate,Labels,Fix Version
"User login flow","See: confluence-link",Story,PROJ-123,2026-01-17,4h,backend,v.Next
"Payment API","See: confluence-link",Story,PROJ-123,2026-01-17,4h,backend,v.Next
"Dashboard UI","See: confluence-link",Story,PROJ-124,2026-01-17,4h,frontend,v.Next
```

**Import:**
1. JIRA → System → External System Import → CSV
2. Map columns
3. Import

**Time:** Create 20 stories in 5 minutes

---

## 3. MAINTENANCE - The Enforcer Fields

### The Most Hateful Fields & How to Handle Them

| Field | Strategy | Default Value | Effort |
|-------|----------|---------------|--------|
| **Due Date** | Pattern-based | Next Friday | 0 thinking |
| **Assignee** | Bulk assign yourself | You | Weekly batch |
| **Estimate** | Always Fibonacci | 4h (always) | 0 thinking |
| **Labels** | 3 rotating labels | backend/frontend/infra | 0 thinking |
| **Release** | One rolling release | v.Next | 0 thinking |

---

### Due Date Strategy

**Never leave blank. Use patterns:**

| Story Type | Formula | Example |
|------------|---------|---------|
| Bug | Today + 3 days | Friday this week |
| Feature | End of sprint | Sprint end date |
| Epic | +3 months | Q2 2026 |
| Spike | +2 weeks | Next Friday |

**Rule:** When in doubt → **Next Friday**

**Bulk edit:**
1. Filter: No due date + Current Sprint
2. Select all
3. Bulk Change → Due Date = (next Friday)
4. Done

---

### Estimate Strategy

**Never think. Use Fibonacci:**

| Story Mentions... | Estimate |
|-------------------|----------|
| "simple", "quick", "small" | 2h |
| **Default** | **4h** |
| "complex", "integration" | 8h |
| "spike", "investigation" | 4h |
| Epic | 40h |

**Rule:** Default to 4h unless title gives clear signal.

**Bulk edit:**
1. Filter: No estimate + Current Sprint
2. Select all
3. Bulk Change → Original Estimate = 4h
4. Done

---

### Assignee Strategy

**Default: Assign to yourself**

**Exceptions:**
- Know it's someone else → Assign immediately
- Don't know → Assign to tech lead + comment "Please assign"
- Infrastructure → Assign to team group

**Bulk edit:**
1. Filter: Assignee = Unassigned + Current Sprint
2. Select all
3. Bulk Change → Assignee = You
4. Done

---

### Labels Strategy

**Create 3 labels, rotate mechanically:**

- `backend`
- `frontend`
- `infrastructure`

**Rule:** Pick based on title keywords:
- API, database, service → `backend`
- UI, button, page → `frontend`
- Everything else → `infrastructure`

**Bulk edit:**
1. Filter: Labels = EMPTY + Current Sprint
2. Select all
3. Bulk Change → Labels = backend (or whatever)
4. Done

---

### Release/Version Strategy

**Create ONE "rolling" release:**

**Setup (Once):**
1. JIRA → Project Settings → Releases
2. Create Release: `v.Next`
3. Description: "Ongoing development"
4. Start Date: Today
5. Release Date: (Leave blank)
6. Save

**Then:** Assign ALL stories to `v.Next`

**Alternative:** Use quarterly releases (Q1 2026, Q2 2026, etc.)

**Bulk edit:**
1. Filter: Fix Version = EMPTY + Current Sprint
2. Select all
3. Bulk Change → Fix Version = v.Next
4. Done

---

## The Weekly "JIRA Hygiene" Routine

**Friday 4:45 PM - 10 Minutes Total**

### Step 1: Create Saved Filter (Once)

**Name:** "Incomplete Stories"

**JQL:**
```jql
project = YOURPROJECT 
AND sprint in openSprints() 
AND (
  assignee is EMPTY 
  OR "due date" is EMPTY 
  OR timeoriginalestimate is EMPTY 
  OR labels is EMPTY 
  OR fixVersion is EMPTY
)
ORDER BY created DESC
```

**Favorite this filter.**

---

### Step 2: Friday Batch Process

**Open "Incomplete Stories" filter → Shows all stories missing enforcer fields**

**Bulk Edit (Select All):**

1. **Assignee** → You (2 min)
2. **Due Date** → Next Friday (2 min)
3. **Original Estimate** → 4h (2 min)
4. **Labels** → backend (2 min)
5. **Fix Version** → v.Next (2 min)

**Total:** 10 minutes, all enforcers satisfied.

---

## Dealing with Specific Enforcers

### The Scrum Master
**Wants:** Estimates, story points filled  
**Give:** Always 4h or 5 story points  
**Defense:** "We validate in sprint retrospective"

### The Product Owner
**Wants:** Due dates aligned with roadmap  
**Give:** Copy sprint end dates  
**Defense:** "Subject to sprint planning adjustments"

### The Engineering Manager
**Wants:** Accurate time tracking  
**Give:** Log 4h per story on completion  
**Defense:** "Reflects focused development time"

### The PMO/Governance
**Wants:** Everything filled, reports look good  
**Give:** Batch-fill weekly with defaults  
**Defense:** "Maintained per process"

---

## Time Logging (If Required)

### Quick Entry Method

**When closing story:**
1. Click story
2. Press `w` (keyboard shortcut for "Log Work")
3. Enter: `4h`
4. Comment: `Development`
5. Save

**Time:** 10 seconds per story

---

### Bulk Time Logging

**If you closed 5 stories this week:**

1. Filter: Status = Done + Updated this week + Assignee = You
2. Select all
3. Bulk Change → Log Work → 4h each
4. Done

**Alternative:** Always log on Friday for everything you touched.

---

## Adding Comments (Least Painful)

### Option 1: Email to JIRA
**If enabled:**
- Send email to: `PROJ-123@your-domain.atlassian.net`
- Body becomes comment

### Option 2: Bulk Comment
1. Write comment in text file
2. Filter stories needing update
3. Bulk Change → Comment
4. Paste same comment to all

### Option 3: Template Comments
**Keep these ready to paste:**
- `Completed as specified`
- `Blocked - waiting on PROJ-456`
- `In progress - 50% complete`
- `Ready for review`

---

## Keyboard Shortcuts (Learn These 5)

| Key | Action | Use |
|-----|--------|-----|
| `g` + `d` | Go to Dashboard | Fast navigation |
| `.` | Quick actions | Open menu without mouse |
| `w` | Log work | Fast time entry |
| `m` | Assign to me | Quick assignment |
| `c` | Create issue | Fast creation |

**Enable:** JIRA → Profile → Settings → Keyboard Shortcuts

---

## Browser Automation (Advanced)

### Auto-Fill New Stories

**Tampermonkey script:**
```javascript
// ==UserScript==
// @name         JIRA Auto-Fill Defaults
// @match        https://your-domain.atlassian.net/*
// ==/UserScript==

(function() {
    'use strict';
    
    // Detect create issue page
    if (window.location.href.includes('CreateIssue')) {
        setTimeout(() => {
            // Due date = 2 weeks from today
            let dueDate = new Date();
            dueDate.setDate(dueDate.getDate() + 14);
            let dueDateStr = dueDate.toISOString().split('T')[0];
            
            let dueDateField = document.querySelector('[name="duedate"]');
            if (dueDateField) dueDateField.value = dueDateStr;
            
            // Estimate = 4h
            let estimateField = document.querySelector('[name="timetracking"]');
            if (estimateField) estimateField.value = '4h';
            
            // Label = backend
            let labelField = document.querySelector('[name="labels"]');
            if (labelField) labelField.value = 'backend';
            
            console.log('JIRA fields auto-filled');
        }, 1500);
    }
})();
```

**Install:** Tampermonkey extension → Create new script → Paste above

**Result:** Bullshit fields auto-fill when creating stories.

---

## Python Helper Scripts

### Bulk Generate Stories CSV

```python
import pandas as pd
from datetime import datetime, timedelta

# Define stories
stories = [
    {"Summary": "User login flow", "Epic": "PROJ-123"},
    {"Summary": "Payment API", "Epic": "PROJ-123"},
    {"Summary": "Dashboard UI", "Epic": "PROJ-124"},
]

# Generate CSV with defaults
df = pd.DataFrame(stories)
df['Description'] = 'See: [confluence-link]'
df['Issue Type'] = 'Story'
df['Due Date'] = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
df['Original Estimate'] = '4h'
df['Labels'] = df['Summary'].apply(lambda x: 'frontend' if 'UI' in x else 'backend')
df['Fix Version'] = 'v.Next'

df.to_csv('jira_bulk_import.csv', index=False)
print(f"Created {len(df)} stories in jira_bulk_import.csv")
```

---

### Extract Comments from Export

```python
import pandas as pd

# Read JIRA CSV export
df = pd.read_csv('jira_export.csv')

# Extract just summaries and comments
for _, row in df.iterrows():
    print(f"\n{'='*60}")
    print(f"{row['Issue key']}: {row['Summary']}")
    print(f"{'='*60}")
    print(f"{row['Comment']}")
```

---

## The Absolute Minimum Workflow

### Daily (30 seconds)
- Drag cards on board: To Do → In Progress → Done
- That's it

### Weekly (10 minutes - Friday 4:45 PM)
1. Open "Incomplete Stories" filter
2. Bulk edit → Fill all defaults
3. Close JIRA
4. Go home

### When Forced to Create Stories (2 minutes per story)
1. Clone template
2. Change title + description
3. Save

**Time spent on JIRA:** ~1 hour/month  
**Time spent on actual work:** Everything else

---

## Checklist: First Time Setup

**Do these once, then forget:**

- [ ] Create template story `PROJ-TEMPLATE`
- [ ] Create release `v.Next` (rolling release)
- [ ] Create 3 labels: `backend`, `frontend`, `infrastructure`
- [ ] Create saved filter "Incomplete Stories" (JQL above)
- [ ] Favorite the filter
- [ ] Enable keyboard shortcuts in profile settings
- [ ] (Optional) Install Tampermonkey auto-fill script
- [ ] Set Friday 4:45 PM calendar reminder: "JIRA Hygiene"

---

## What to Do When Enforcers Complain

### "Your estimates are always 4h!"
**Response:** "We're using relative sizing. 4h represents medium complexity. Validated in retro."

### "Your due dates are all the same!"
**Response:** "Aligned with sprint cadence. Subject to daily standup adjustments."

### "You're not logging time accurately!"
**Response:** "Logging focused development hours. Matches velocity tracking."

### "Your labels are too generic!"
**Response:** "Using team taxonomy. Maps to our component architecture."

### "Everything is in v.Next!"
**Response:** "Continuous delivery model. Version assigned at release time."

**Key:** Sound confident. Use corporate jargon. They'll move on.

---

## Emergency: "Manager Wants Detailed Updates"

**If forced to provide detail:**

**Template Response:**
```
Story: PROJ-123
Status: In Progress (60% complete)
Blockers: None
Next Steps: Complete testing, submit PR
ETA: End of sprint
```

**Copy-paste this, change percentages randomly.**

**They want to see progress. Give them numbers. Move on.**

---

## Success Metrics

**You're doing it right if:**
- ✅ Spend <1 hour/month in JIRA
- ✅ No enforcer emails about missing fields
- ✅ All stories have required fields filled
- ✅ Zero thinking about JIRA during actual work
- ✅ Can batch-process everything Friday afternoon

**You're doing it wrong if:**
- ❌ Spending >10 min/day in JIRA
- ❌ Thinking carefully about estimates
- ❌ Custom due dates per story
- ❌ Detailed time tracking
- ❌ Reading JIRA reports

---

## Remember

**JIRA is not your work. JIRA is compliance theater.**

**Your real work:** Code, design, architecture, problem-solving

**JIRA:** Minimum viable compliance to satisfy enforcers

**10 minutes Friday = satisfied management, zero interruptions rest of week**

---

## Quick Reference Card (Print This)

```
┌─────────────────────────────────────────┐
│     JIRA DEFAULTS - NEVER THINK         │
├─────────────────────────────────────────┤
│ Due Date:     Next Friday               │
│ Estimate:     4h                        │
│ Assignee:     You                       │
│ Label:        backend                   │
│ Release:      v.Next                    │
├─────────────────────────────────────────┤
│ FRIDAY 4:45 PM - BULK EDIT ALL          │
├─────────────────────────────────────────┤
│ Filter: "Incomplete Stories"            │
│ Select All → Fill Defaults → Done      │
└─────────────────────────────────────────┘
```

---

**Good luck. May your JIRA time be minimal and your actual work time be maximal.**