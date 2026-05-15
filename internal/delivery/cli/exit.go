package cli

// ExitCodeFor maps an error to a POSIX exit code per the spec table.
// Implementation in L5 covers all cases; this stub keeps the build green.
func ExitCodeFor(err error) int {
	if err == nil {
		return 0
	}
	return 1
}
