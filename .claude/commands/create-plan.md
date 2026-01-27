# Create Implementation Plan

You are tasked with creating detailed implementation plans through an interactive, iterative process. Be skeptical, thorough, and work collaboratively with the user to produce high-quality technical specifications.

## Initial Response

When this command is invoked:

1. **If a file path or description was provided as a parameter**:
   - Read any provided files FULLY
   - Begin the research process immediately

2. **If no parameters provided**, respond with:
   ```
   I'll help you create a detailed implementation plan. Let me start by understanding what we're building.

   Please provide:
   1. The task description or feature requirements
   2. Any relevant context, constraints, or specific requirements
   3. Links to related code or previous implementations

   Tip: You can invoke this with a description: `/create_plan Add DEM channel support to the segmentation model`
   ```
   Then wait for user input.

## Process Steps

### Step 1: Context Gathering & Initial Analysis

1. **Read all mentioned files immediately and FULLY**:
   - Requirements or design documents
   - Related source files
   - Configuration files
   - **IMPORTANT**: Use the Read tool WITHOUT limit/offset parameters
   - **CRITICAL**: Read files yourself before spawning research tasks

2. **Research the codebase**:
   Use the Task tool with `subagent_type=Explore` to research in parallel:
   - Find all files related to the task
   - Understand how current implementation works
   - Identify patterns and conventions to follow

3. **Read all files identified by research**:
   - After research completes, read ALL relevant files FULLY
   - This ensures complete understanding before proceeding

4. **Analyze and verify understanding**:
   - Cross-reference requirements with actual code
   - Identify discrepancies or misunderstandings
   - Note assumptions that need verification

5. **Present informed understanding and focused questions**:
   ```
   Based on my research of the codebase, I understand we need to [accurate summary].

   I've found that:
   - [Current implementation detail with file:line reference]
   - [Relevant pattern or constraint discovered]
   - [Potential complexity or edge case identified]

   Questions that my research couldn't answer:
   - [Specific technical question requiring human judgment]
   - [Design preference that affects implementation]
   ```

### Step 2: Research & Discovery

After getting initial clarifications:

1. **If the user corrects any misunderstanding**:
   - Spawn new research tasks to verify the correct information
   - Read the specific files/directories they mention
   - Only proceed once verified

2. **Spawn parallel research tasks** using Task tool with Explore agent:
   - Research different aspects concurrently
   - Find similar implementations to model after
   - Identify integration points and dependencies

3. **Wait for ALL research to complete** before proceeding

4. **Present findings and design options**:
   ```
   Based on my research, here's what I found:

   **Current State:**
   - [Key discovery about existing code]
   - [Pattern or convention to follow]

   **Design Options:**
   1. [Option A] - [pros/cons]
   2. [Option B] - [pros/cons]

   **Open Questions:**
   - [Technical uncertainty]
   - [Design decision needed]

   Which approach aligns best with your vision?
   ```

### Step 3: Plan Structure Development

Once aligned on approach:

1. **Create initial plan outline**:
   ```
   Here's my proposed plan structure:

   ## Overview
   [1-2 sentence summary]

   ## Implementation Phases:
   1. [Phase name] - [what it accomplishes]
   2. [Phase name] - [what it accomplishes]
   3. [Phase name] - [what it accomplishes]

   Does this phasing make sense? Should I adjust the order or granularity?
   ```

2. **Get feedback on structure** before writing details

### Step 4: Detailed Plan Writing

After structure approval:

1. **Write the plan** to `docs/plans/YYYY-MM-DD-description.md`
   - Format: `YYYY-MM-DD-description.md`
   - Example: `2026-01-27-dem-channel-support.md`

2. **Use this template structure**:

````markdown
# [Feature/Task Name] Implementation Plan

## Overview

[Brief description of what we're implementing and why]

## Current State Analysis

[What exists now, what's missing, key constraints discovered]

### Key Discoveries:
- [Important finding with file:line reference]
- [Pattern to follow]
- [Constraint to work within]

## Desired End State

[Specification of the desired end state and how to verify it]

## What We're NOT Doing

[Explicitly list out-of-scope items to prevent scope creep]

## Implementation Approach

[High-level strategy and reasoning]

---

## Phase 1: [Descriptive Name]

### Overview
[What this phase accomplishes]

### Changes Required:

#### 1. [Component/File Group]
**File**: `path/to/file.py`
**Changes**: [Summary of changes]

```python
# Specific code to add/modify
```

### Success Criteria:

#### Automated Verification:
- [ ] Linting passes: `uv run task lint`
- [ ] Tests pass: `uv run task test`
- [ ] Type checking passes: `uv run mypy src/`

#### Manual Verification:
- [ ] Feature works as expected when tested
- [ ] No regressions in related features

**Implementation Note**: After completing this phase and all automated verification passes, pause for manual confirmation before proceeding to the next phase.

---

## Phase 2: [Descriptive Name]

[Similar structure...]

---

## Testing Strategy

### Unit Tests:
- [What to test]
- [Key edge cases]

### Integration Tests:
- [End-to-end scenarios]

## References

- Related code: `[file:line]`
- Documentation: `docs/[relevant].md`
````

### Step 5: Review

1. **Present the draft plan location**:
   ```
   I've created the implementation plan at:
   `docs/plans/YYYY-MM-DD-description.md`

   Please review it and let me know:
   - Are the phases properly scoped?
   - Are the success criteria specific enough?
   - Any technical details that need adjustment?
   ```

2. **Iterate based on feedback** until satisfied

## Important Guidelines

1. **Be Skeptical**:
   - Question vague requirements
   - Identify potential issues early
   - Don't assume - verify with code

2. **Be Interactive**:
   - Don't write the full plan in one shot
   - Get buy-in at each major step
   - Allow course corrections

3. **Be Thorough**:
   - Read all context files COMPLETELY
   - Research actual code patterns using Task tool
   - Include specific file paths and line numbers
   - Write measurable success criteria

4. **Be Practical**:
   - Focus on incremental, testable changes
   - Consider edge cases
   - Include "what we're NOT doing"

5. **No Open Questions in Final Plan**:
   - If you encounter open questions, STOP
   - Research or ask for clarification immediately
   - The plan must be complete and actionable

## Success Criteria Guidelines

**Always separate into two categories:**

1. **Automated Verification**:
   - `uv run task lint` - linting and formatting
   - `uv run task test` - tests with coverage
   - Specific commands that verify functionality

2. **Manual Verification**:
   - Visual inspection of outputs
   - Performance under real conditions
   - Edge cases that are hard to automate
