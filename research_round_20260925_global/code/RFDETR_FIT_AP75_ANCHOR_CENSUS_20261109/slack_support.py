"""Integer IoU=.75 slack for area-preserving 4+4 pixel edits."""
def slack_after(native_slack,hits):
    return native_slack+7*(hits-4)


def needs_hits(native_slack):
    return max(0,4-native_slack//7)


def bounds(correct,inside):
    lo=hi=0
    for side in (inside,~inside):
        n=int(side.sum())
        if n<4:
            return None
        pos=int(correct[side].sum())
        lo+=max(0,4-(n-pos))
        hi+=min(4,pos)
    return lo,hi
