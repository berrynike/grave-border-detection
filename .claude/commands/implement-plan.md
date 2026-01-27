# Implement Plan

You are tasked with implementing an approved technical plan from `docs/plans/`. These plans contain phases with specific changes and success criteria.

## Getting Started

When given a plan path:
- Read the plan completely and check for any existing checkmarks (`- [x]`)
- Read all files mentioned in the plan FULLY
- Think deeply about how the pieces fit together
- Create a todo list to track your progress
- Start implementing if you understand what needs to be done

If no plan path provided, ask for one:
```
Which plan would you like to implement? Please provide the path.

Tip: List recent plans with `ls -lt docs/plans/ | head`
```

## Implementation Philosophy

Plans are carefully designed, but reality can be messy. Your job is to:
- Follow the plan's intent while adapting to what you find
- Implement each phase fully before moving to the next
- Verify your work makes sense in the broader codebase context
- Update checkboxes in the plan as you complete sections

When things don't match the plan exactly, think about why and communicate clearly.

## Handling Mismatches

If you encounter a mismatch between the plan and reality:

1. **STOP** and think deeply about why the plan can't be followed
2. **Present the issue clearly**:
   ```
   Issue in Phase [N]:
   Expected: [what the plan says]
   Found: [actual situation]
   Why this matters: [explanation]

   How should I proceed?
   ```
3. Wait for guidance before continuing

## Verification Approach

After implementing each phase:

1. **Run automated verification**:
   ```bash
   uv run task lint   # Linting, formatting, type checking
   uv run task test   # Tests with coverage
   ```

2. **Fix any issues** before proceeding

3. **Update progress**:
   - Check off completed items in the plan file using Edit
   - Update your todo list

4. **Pause for manual verification**:
   ```
   Phase [N] Complete - Ready for Manual Verification

   Automated verification passed:
   - [List automated checks that passed]

   Please perform the manual verification steps listed in the plan:
   - [List manual verification items from the plan]

   Let me know when manual testing is complete so I can proceed to Phase [N+1].
   ```

**Important**: Do NOT check off manual verification items until confirmed by the user.

If instructed to execute multiple phases consecutively, skip the pause until the last phase.

## If You Get Stuck

When something isn't working as expected:

1. Make sure you've read and understood all the relevant code
2. Consider if the codebase has evolved since the plan was written
3. Use Task tool with Explore agent for targeted investigation
4. Present the mismatch clearly and ask for guidance

## Resuming Work

If the plan has existing checkmarks:
- Trust that completed work is done
- Pick up from the first unchecked item
- Verify previous work only if something seems off

## Key Reminders

- **Read files FULLY** - never use limit/offset parameters
- **Follow project conventions** - check `CLAUDE.md` and `docs/python-rules.md`
- **Preserve CRS** in all geospatial operations
- **Run lint and test** after each phase
- **No silent fallbacks** - let errors fail loudly
- Keep the end goal in mind and maintain forward momentum
